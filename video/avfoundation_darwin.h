#include <stdint.h>

typedef struct {
	int frames;
	double duration;
	int width;  // coded size, before rotation
	int height;
	int rotation; // clockwise degrees to display: 0, 90, 180 or 270
	int audio;
	int sample_rate;
	int channels;
} ov_meta;

// ov_open loads the asset's metadata, reading the file as the container
// type mime. It returns an opaque handle, or NULL with *err set to a
// malloc'd message. Tracks transformed by anything but a quarter-turn
// rotation are rejected.
void *ov_open(const char *path, const char *mime, ov_meta *meta, char **err);
void ov_close(void *asset);

// ov_frames_begin starts decoding the video track in presentation order.
void *ov_frames_begin(void *asset, char **err);

// ov_frames_next decodes the next frame. When want is set, it copies the
// BGRA pixels into a malloc'd *pix. It returns 1 for a frame, 0 at the end,
// and -1 on error with *err set.
int ov_frames_next(void *reader, int want, uint8_t **pix, int *width, int *height, int *stride, char **err);
void ov_frames_end(void *reader);

// ov_pcm decodes the first audio track, of the given channel count, as float32
// at its native rate, downmixed to mono into a malloc'd *samples. It returns
// -2 as soon as the track holds more than max_frames samples.
int ov_pcm(void *asset, int channels, int64_t max_frames, float **samples, int64_t *count, char **err);
