# Video decoder fixtures

Synthetic 32 x 32, 2 fps clips: two red frames, then two blue frames. These contain
no audio, except `red-blue-audio.mp4`, which adds a synthetic 880 Hz AAC
soundtrack. `red.webp` is a still from the first frame. MP4 files deliberately keep metadata at the end to exercise streaming
input through `cache:pipe:0`. H.264 and HEVC use reordered frames.

Created from four RGB24 frames using FFmpeg, with `-c:v libx264 -bf 2`,
`-c:v libx265`, or `-c:v libvpx-vp9`, and `-pix_fmt yuv420p`.
No external media or licensed content is included.
