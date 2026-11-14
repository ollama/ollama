#import <AVFoundation/AVFoundation.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>
#include <stdlib.h>
#include <string.h>

#include "avfoundation_darwin.h"

static char *ov_error(NSString *what, NSError *error) {
	NSString *msg = error ? [NSString stringWithFormat:@"%@: %@", what, error.localizedDescription] : what;
	return strdup(msg.UTF8String);
}

static char *ov_exception(NSException *e) {
	return strdup([NSString stringWithFormat:@"AVFoundation: %@", e.reason].UTF8String);
}

// Each exported entry point runs one of these inside @try.
static void *ov_open_asset(const char *path, const char *mime, ov_meta *meta, char **err);
static void *ov_frames_start(void *handle, char **err);
static int ov_frame(void *handle, int want, uint8_t **pix, int *width, int *height, int *stride, char **err);
static int ov_audio(void *handle, int channels, int64_t max_frames, float **samples, int64_t *count, char **err);

// ov_rotation returns the clockwise quarter turns a track transform
// displays with, or -1 for any other transform (mirror, skew, scale).
static int ov_rotation(CGAffineTransform t) {
	if (t.a == 1 && t.b == 0 && t.c == 0 && t.d == 1) {
		return 0;
	}
	if (t.a == 0 && t.b == 1 && t.c == -1 && t.d == 0) {
		return 90;
	}
	if (t.a == -1 && t.b == 0 && t.c == 0 && t.d == -1) {
		return 180;
	}
	if (t.a == 0 && t.b == -1 && t.c == 1 && t.d == 0) {
		return 270;
	}
	return -1;
}

// ov_tracks blocks until the asset's tracks of a media type have loaded.
static NSArray<AVAssetTrack *> *ov_tracks(AVAsset *asset, AVMediaType type, NSError **error) {
	__block NSArray<AVAssetTrack *> *tracks = nil;
	__block NSError *loadError = nil;
	dispatch_semaphore_t done = dispatch_semaphore_create(0);
	[asset loadTracksWithMediaType:type
	             completionHandler:^(NSArray<AVAssetTrack *> *result, NSError *e) {
		             tracks = result;
		             loadError = e;
		             dispatch_semaphore_signal(done);
	             }];
	dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
	if (error) {
		*error = loadError;
	}
	return tracks;
}

// ov_load blocks until the given track keys have loaded.
static BOOL ov_load(AVAssetTrack *track, NSArray<NSString *> *keys, NSError **error) {
	dispatch_semaphore_t done = dispatch_semaphore_create(0);
	[track loadValuesAsynchronouslyForKeys:keys
	                     completionHandler:^{
		                     dispatch_semaphore_signal(done);
	                     }];
	dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
	for (NSString *key in keys) {
		if ([track statusOfValueForKey:key error:error] != AVKeyValueStatusLoaded) {
			return NO;
		}
	}
	return YES;
}

void *ov_open(const char *path, const char *mime, ov_meta *meta, char **err) {
	@autoreleasepool {
		@try {
			return ov_open_asset(path, mime, meta, err);
		} @catch (NSException *e) {
			*err = ov_exception(e);
			return NULL;
		}
	}
}

static void *ov_open_asset(const char *path, const char *mime, ov_meta *meta, char **err) {
	{
		NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
		// The container type comes from the content, so files need no extension.
		NSDictionary *options = @{
			AVURLAssetPreferPreciseDurationAndTimingKey : @YES,
			AVURLAssetOverrideMIMETypeKey : [NSString stringWithUTF8String:mime],
		};
		AVURLAsset *asset = [AVURLAsset URLAssetWithURL:url options:options];
		NSError *error = nil;
		NSArray<AVAssetTrack *> *video = ov_tracks(asset, AVMediaTypeVideo, &error);
		if (video.count == 0) {
			*err = ov_error(@"no video track", error);
			return NULL;
		}
		AVAssetTrack *track = video[0];
		if (!ov_load(track, @[ @"timeRange", @"formatDescriptions", @"preferredTransform" ], &error)) {
			*err = ov_error(@"loading video track", error);
			return NULL;
		}
		memset(meta, 0, sizeof(*meta));
		meta->duration = CMTimeGetSeconds(track.timeRange.duration);
		// The codec's size is what the decoder outputs; the track header's
		// naturalSize can differ (pixel aspect, or a corrupt header).
		CMFormatDescriptionRef format = (__bridge CMFormatDescriptionRef)track.formatDescriptions.firstObject;
		if (format) {
			CMVideoDimensions dims = CMVideoFormatDescriptionGetDimensions(format);
			meta->width = dims.width;
			meta->height = dims.height;
		}

		meta->rotation = ov_rotation(track.preferredTransform);
		if (meta->rotation < 0) {
			*err = ov_error(@"unsupported video transform (only quarter-turn rotations)", nil);
			return NULL;
		}

		// Count samples without decoding; the frame count and duration give
		// the average rate, as the reference processor's decoder reports it.
		if (track.canProvideSampleCursors) {
			AVSampleCursor *cursor = [track makeSampleCursorAtFirstSampleInDecodeOrder];
			meta->frames = cursor ? (int)[cursor stepInDecodeOrderByCount:INT32_MAX] + 1 : 0;
		}

		NSArray<AVAssetTrack *> *audio = ov_tracks(asset, AVMediaTypeAudio, &error);
		if (audio.count > 0 && ov_load(audio[0], @[ @"formatDescriptions" ], &error)) {
			CMFormatDescriptionRef desc = (__bridge CMFormatDescriptionRef)audio[0].formatDescriptions.firstObject;
			const AudioStreamBasicDescription *asbd = desc ? CMAudioFormatDescriptionGetStreamBasicDescription(desc) : NULL;
			if (asbd && asbd->mSampleRate > 0 && asbd->mChannelsPerFrame > 0) {
				meta->audio = 1;
				meta->sample_rate = (int)asbd->mSampleRate;
				meta->channels = (int)asbd->mChannelsPerFrame;
			}
		}
		return (void *)CFBridgingRetain(asset);
	}
}

void ov_close(void *asset) {
	if (asset) {
		CFBridgingRelease(asset);
	}
}

typedef struct {
	void *reader; // AVAssetReader, retained
	void *output; // AVAssetReaderTrackOutput, retained
} ov_reader;

void *ov_frames_begin(void *handle, char **err) {
	@autoreleasepool {
		@try {
			return ov_frames_start(handle, err);
		} @catch (NSException *e) {
			*err = ov_exception(e);
			return NULL;
		}
	}
}

static void *ov_frames_start(void *handle, char **err) {
	{
		AVAsset *asset = (__bridge AVAsset *)handle;
		NSError *error = nil;
		NSArray<AVAssetTrack *> *video = ov_tracks(asset, AVMediaTypeVideo, &error);
		if (video.count == 0) {
			*err = ov_error(@"no video track", error);
			return NULL;
		}
		AVAssetReader *reader = [AVAssetReader assetReaderWithAsset:asset error:&error];
		if (!reader) {
			*err = ov_error(@"creating reader", error);
			return NULL;
		}
		NSDictionary *settings = @{(id)kCVPixelBufferPixelFormatTypeKey : @(kCVPixelFormatType_32BGRA)};
		AVAssetReaderTrackOutput *output = [AVAssetReaderTrackOutput assetReaderTrackOutputWithTrack:video[0] outputSettings:settings];
		output.alwaysCopiesSampleData = NO;
		if (![reader canAddOutput:output]) {
			*err = ov_error(@"reader rejected the video output", nil);
			return NULL;
		}
		[reader addOutput:output];
		if (![reader startReading]) {
			*err = ov_error(@"starting reader", reader.error);
			return NULL;
		}
		ov_reader *r = calloc(1, sizeof(ov_reader));
		if (!r) {
			[reader cancelReading];
			*err = ov_error(@"out of memory", nil);
			return NULL;
		}
		r->reader = (void *)CFBridgingRetain(reader);
		r->output = (void *)CFBridgingRetain(output);
		return r;
	}
}

int ov_frames_next(void *handle, int want, uint8_t **pix, int *width, int *height, int *stride, char **err) {
	@autoreleasepool {
		@try {
			return ov_frame(handle, want, pix, width, height, stride, err);
		} @catch (NSException *e) {
			*err = ov_exception(e);
			return -1;
		}
	}
}

static int ov_frame(void *handle, int want, uint8_t **pix, int *width, int *height, int *stride, char **err) {
	{
		ov_reader *r = handle;
		AVAssetReader *reader = (__bridge AVAssetReader *)r->reader;
		AVAssetReaderTrackOutput *output = (__bridge AVAssetReaderTrackOutput *)r->output;
		for (;;) {
			CMSampleBufferRef sample = [output copyNextSampleBuffer];
			if (!sample) {
				if (reader.status == AVAssetReaderStatusFailed) {
					*err = ov_error(@"decoding video", reader.error);
					return -1;
				}
				return 0;
			}
			CVImageBufferRef image = CMSampleBufferGetImageBuffer(sample);
			if (!image) { // a marker sample without pixels: not a frame
				CFRelease(sample);
				continue;
			}
			if (want) {
				CVPixelBufferLockBaseAddress(image, kCVPixelBufferLock_ReadOnly);
				size_t w = CVPixelBufferGetWidth(image), h = CVPixelBufferGetHeight(image);
				size_t rowBytes = CVPixelBufferGetBytesPerRow(image);
				*pix = malloc(rowBytes * h);
				if (*pix) {
					memcpy(*pix, CVPixelBufferGetBaseAddress(image), rowBytes * h);
				}
				CVPixelBufferUnlockBaseAddress(image, kCVPixelBufferLock_ReadOnly);
				if (!*pix) {
					CFRelease(sample);
					*err = ov_error(@"out of memory", nil);
					return -1;
				}
				*width = (int)w;
				*height = (int)h;
				*stride = (int)rowBytes;
			}
			CFRelease(sample);
			return 1;
		}
	}
}

void ov_frames_end(void *handle) {
	ov_reader *r = handle;
	if (!r) {
		return;
	}
	AVAssetReader *reader = CFBridgingRelease(r->reader);
	[reader cancelReading];
	CFBridgingRelease(r->output);
	free(r);
}

int ov_pcm(void *handle, int channels, int64_t max_frames, float **samples, int64_t *count, char **err) {
	@autoreleasepool {
		@try {
			return ov_audio(handle, channels, max_frames, samples, count, err);
		} @catch (NSException *e) {
			*err = ov_exception(e);
			return -1;
		}
	}
}

static int ov_audio(void *handle, int channels, int64_t max_frames, float **samples, int64_t *count, char **err) {
	{
		AVAsset *asset = (__bridge AVAsset *)handle;
		NSError *error = nil;
		NSArray<AVAssetTrack *> *audio = ov_tracks(asset, AVMediaTypeAudio, &error);
		*samples = NULL;
		*count = 0;
		if (audio.count == 0) {
			return 0;
		}
		AVAssetReader *reader = [AVAssetReader assetReaderWithAsset:asset error:&error];
		if (!reader) {
			*err = ov_error(@"creating reader", error);
			return -1;
		}
		NSDictionary *settings = @{
			AVFormatIDKey : @(kAudioFormatLinearPCM),
			AVLinearPCMBitDepthKey : @32,
			AVLinearPCMIsFloatKey : @YES,
			AVLinearPCMIsBigEndianKey : @NO,
			AVLinearPCMIsNonInterleaved : @NO,
		};
		AVAssetReaderTrackOutput *output = [AVAssetReaderTrackOutput assetReaderTrackOutputWithTrack:audio[0] outputSettings:settings];
		if (![reader canAddOutput:output]) {
			*err = ov_error(@"reader rejected the audio output", nil);
			return -1;
		}
		[reader addOutput:output];
		if (![reader startReading]) {
			*err = ov_error(@"starting reader", reader.error);
			return -1;
		}
		// Each buffer is downmixed as it decodes, so memory follows the mono
		// length whatever channel count the container declares.
		NSMutableData *scratch = [NSMutableData data];
		float *mono = NULL;
		int64_t frames = 0, capacity = 0;
		int status = 0;
		for (;;) {
			@autoreleasepool {
				CMSampleBufferRef sample = [output copyNextSampleBuffer];
				if (!sample) {
					break;
				}
				CMBlockBufferRef block = CMSampleBufferGetDataBuffer(sample);
				size_t length = block ? CMBlockBufferGetDataLength(block) : 0;
				int64_t n = (int64_t)(length / sizeof(float) / channels);
				// The track can outlast the video, so its length is checked as it decodes.
				if (frames + n > max_frames) {
					status = -2;
				} else if (n > 0 && frames + n > capacity) {
					capacity = MIN(MAX(frames + n, 2 * capacity), max_frames);
					float *grown = realloc(mono, capacity * sizeof(float));
					if (grown) {
						mono = grown;
					} else {
						*err = ov_error(@"out of memory", nil);
						status = -1;
					}
				}
				if (status == 0 && n > 0) {
					[scratch setLength:length];
					CMBlockBufferCopyDataBytes(block, 0, length, scratch.mutableBytes);
					const float *in = scratch.bytes;
					for (int64_t i = 0; i < n; i++) {
						float sum = 0;
						for (int c = 0; c < channels; c++) {
							sum += in[i * channels + c];
						}
						mono[frames + i] = sum / (float)channels;
					}
					frames += n;
				}
				CFRelease(sample);
			}
			if (status != 0) {
				[reader cancelReading];
				free(mono);
				return status;
			}
		}
		if (reader.status == AVAssetReaderStatusFailed) {
			free(mono);
			*err = ov_error(@"decoding audio", reader.error);
			return -1;
		}
		*samples = mono;
		*count = frames;
		return 0;
	}
}
