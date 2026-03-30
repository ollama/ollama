#import <AVFoundation/AVFoundation.h>
#import <CoreGraphics/CoreGraphics.h>
#import <CoreMedia/CoreMedia.h>
#import <ImageIO/ImageIO.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>
#import <Foundation/Foundation.h>
#include <stdlib.h>
#include <stdint.h>

void free_ptr(void* p) {
    free(p);
}

// Convert a CGImage to JPEG data.
static NSData* cgImageToJPEG(CGImageRef image) {
    NSMutableData *data = [NSMutableData data];
    CGImageDestinationRef dest = CGImageDestinationCreateWithData(
        (__bridge CFMutableDataRef)data, (__bridge CFStringRef)UTTypeJPEG.identifier, 1, NULL);
    if (!dest) return nil;

    NSDictionary *props = @{(__bridge NSString *)kCGImageDestinationLossyCompressionQuality: @(0.85)};
    CGImageDestinationAddImage(dest, image, (__bridge CFDictionaryRef)props);
    CGImageDestinationFinalize(dest);
    CFRelease(dest);
    return data;
}

int extract_video_frames(
    const char* path,
    int max_frames,
    int extract_audio,
    uint8_t** frame_data,
    int* frame_sizes,
    int* num_frames,
    uint8_t** audio_data,
    int* audio_size)
{
    @autoreleasepool {
        *num_frames = 0;
        *audio_size = 0;
        *audio_data = NULL;

        NSString *filePath = [NSString stringWithUTF8String:path];
        NSURL *fileURL = [NSURL fileURLWithPath:filePath];
        AVURLAsset *asset = [AVURLAsset URLAssetWithURL:fileURL options:nil];

        // Get video duration
        CMTime duration = asset.duration;
        if (CMTIME_IS_INVALID(duration) || CMTimeGetSeconds(duration) <= 0) {
            return -1;
        }
        Float64 durationSecs = CMTimeGetSeconds(duration);

        // Create image generator
        AVAssetImageGenerator *generator = [[AVAssetImageGenerator alloc]
            initWithAsset:asset];
        generator.appliesPreferredTrackTransform = YES;
        generator.requestedTimeToleranceBefore = kCMTimeZero;
        generator.requestedTimeToleranceAfter = kCMTimeZero;

        // Calculate frame times evenly spaced across duration
        int frameCount = max_frames;
        if (durationSecs < frameCount) {
            frameCount = (int)durationSecs;
        }
        if (frameCount < 1) frameCount = 1;

        // Extract frames using synchronous API.
        // Note: copyCGImageAtTime: is deprecated in macOS 15 in favor of the
        // async generateCGImagesAsynchronouslyForTimes:, but the async API
        // is incompatible with CGo (callbacks on arbitrary threads). The sync
        // API remains functional and is the safest approach for CGo callers.
        int extracted = 0;
        for (int i = 0; i < frameCount; i++) {
            Float64 t = (durationSecs * i) / frameCount;
            CMTime requestTime = CMTimeMakeWithSeconds(t, 600);
            CMTime actualTime;
            NSError *error = nil;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            CGImageRef cgImage = [generator copyCGImageAtTime:requestTime
                                                   actualTime:&actualTime
                                                        error:&error];
#pragma clang diagnostic pop
            if (!cgImage) continue;

            NSData *jpegData = cgImageToJPEG(cgImage);
            CGImageRelease(cgImage);

            if (!jpegData || jpegData.length == 0) continue;

            uint8_t *buf = (uint8_t *)malloc(jpegData.length);
            memcpy(buf, jpegData.bytes, jpegData.length);
            frame_data[extracted] = buf;
            frame_sizes[extracted] = (int)jpegData.length;
            extracted++;
        }
        *num_frames = extracted;

        // Extract audio if requested.
        if (extract_audio) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            NSArray<AVAssetTrack *> *audioTracks =
                [asset tracksWithMediaType:AVMediaTypeAudio];
#pragma clang diagnostic pop

            if (audioTracks.count > 0) {
                NSError *error = nil;
                AVAssetReader *reader = [[AVAssetReader alloc]
                    initWithAsset:asset error:&error];
                if (reader) {
                    NSDictionary *settings = @{
                        AVFormatIDKey: @(kAudioFormatLinearPCM),
                        AVSampleRateKey: @(16000),
                        AVNumberOfChannelsKey: @(1),
                        AVLinearPCMBitDepthKey: @(16),
                        AVLinearPCMIsFloatKey: @(NO),
                        AVLinearPCMIsBigEndianKey: @(NO),
                    };
                    AVAssetReaderTrackOutput *output =
                        [[AVAssetReaderTrackOutput alloc]
                            initWithTrack:audioTracks[0]
                            outputSettings:settings];
                    [reader addOutput:output];

                    if ([reader startReading]) {
                        NSMutableData *pcmData = [NSMutableData data];
                        CMSampleBufferRef sampleBuffer;
                        while ((sampleBuffer = [output copyNextSampleBuffer])) {
                            CMBlockBufferRef blockBuffer =
                                CMSampleBufferGetDataBuffer(sampleBuffer);
                            size_t length = CMBlockBufferGetDataLength(blockBuffer);
                            uint8_t *tmp = (uint8_t *)malloc(length);
                            CMBlockBufferCopyDataBytes(blockBuffer, 0, length, tmp);
                            [pcmData appendBytes:tmp length:length];
                            free(tmp);
                            CFRelease(sampleBuffer);
                        }

                        if (pcmData.length > 0) {
                            *audio_data = (uint8_t *)malloc(pcmData.length);
                            memcpy(*audio_data, pcmData.bytes, pcmData.length);
                            *audio_size = (int)pcmData.length;
                        }
                    }
                }
            }
        }

        return 0;
    }
}
