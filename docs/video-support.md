# Video Support in Ollama

Ollama supports video understanding through multimodal models like Qwen3-VL. This document explains how video support works and how to use it.

## Overview

Video support enables models to analyze and understand video content by:
1. Extracting frames from videos using FFmpeg
2. Processing frames with temporal awareness
3. Understanding the sequence and motion in the video

## Supported Models

Currently, the following models support video understanding:
- **qwen3-vl** - Qwen's 3rd generation vision-language model with native video support (Interleaved-MRoPE position embeddings)
- **qwen2.5-vl** - Qwen's 2.5 generation vision-language model with video support (standard RoPE position embeddings)

## Requirements

### System Requirements
- **FFmpeg**: Must be installed and available in your system PATH
  - Ubuntu/Debian: `sudo apt-get install ffmpeg`
  - macOS: `brew install ffmpeg`
  - Windows: `choco install ffmpeg`

### Supported Video Formats
- MP4 (recommended)
- MOV
- AVI
- MKV
- WebM
- M4V

Any format supported by FFmpeg can be processed.

## API Usage

### Generate API

Send videos as base64-encoded data in the `videos` parameter:

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "qwen3-vl",
  "prompt": "What is happening in this video?",
  "videos": ["<base64-encoded-video>"]
}'

# Or use qwen2.5-vl
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5-vl",
  "prompt": "What is happening in this video?",
  "videos": ["<base64-encoded-video>"]
}'
```

**Example with actual video:**

```bash
# Encode video to base64
VIDEO_BASE64=$(base64 -i video.mp4)

# Send request
curl http://localhost:11434/api/generate -d "{
  \"model\": \"qwen3-vl\",
  \"prompt\": \"Describe what you see in this video.\",
  \"videos\": [\"$VIDEO_BASE64\"]
}"
```

### Chat API

Videos can be included in chat messages:

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "qwen3-vl",
  "messages": [
    {
      "role": "user",
      "content": "What happens in this video?",
      "videos": ["<base64-encoded-video>"]
    }
  ]
}'
```

### OpenAI Compatible API

The OpenAI-compatible endpoint supports videos through the `video_url` content type:

```bash
curl http://localhost:11434/v1/chat/completions -d '{
  "model": "qwen3-vl",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "What is in this video?"},
      {"type": "video_url", "video_url": {"url": "https://example.com/video.mp4"}}
    ]
  }]
}'
```

Videos can be provided as:
- **HTTP/HTTPS URLs**: The video will be downloaded
- **Base64 data URLs**: `data:video/mp4;base64,<base64-data>`

## Desktop App

The Ollama desktop app supports video uploads in the chat interface.

### Uploading Videos

1. Click the attachment button in the chat interface
2. Select a video file (MP4, MOV, AVI, MKV, WebM, M4V)
3. The video will be uploaded and processed
4. Ask questions about the video content

## Technical Architecture

### Frame Extraction

Videos are processed through the following pipeline:

1. **Video Detection**: Check if input is a video (not an image)
2. **Frame Extraction**: Use FFmpeg to extract frames at 1.0 FPS (configurable)
3. **Frame Processing**: Each frame is:
   - Composited (alpha channel removed)
   - Resized using smart resizing
   - Normalized with ImageNet statistics

Same process could be adapted to allow for simulated video processing with frames in non-video supported vision models but they would lack temporal (time) understanding and yeild unfavourable results.

### Temporal Processing

Temporal processing approach in supported models such as Qwen3-vl:

- **Temporal Patches**: Frames are grouped using `temporalPatchSize=2` (pairs of consecutive frames)
- **3D Grid Structure**: Tracks dimensions as Temporal × Height × Width
- **Position Embeddings**: 4D position embeddings (time, height, width, extra dimension)
- **Conv3D Processing**: 3D convolution handles temporal dimension natively

### Single-Frame Handling

Videos with only 1 frame are automatically processed as static images for efficiency and compatibility.

## Configuration

### Frame Extraction Settings

Frame extraction can be configured programmatically (not exposed via API yet):

```go
config := imageproc.VideoExtractionConfig{
    FPS:       1.0,              // Frames per second to extract
    Quality:   2,                // JPEG quality (1-31, lower is better)
    MaxFrames: 0,                // Maximum frames (0 = no limit)
    Timeout:   60 * time.Second, // Extraction timeout
}
```

Default settings:
- **FPS**: 1.0 (one frame per second)
- **Quality**: 2 (high quality)
- **MaxFrames**: 0 (no limit)
- **Timeout**: 60 seconds

## Performance Considerations

### Video Length

- **Short videos** (< 10 seconds): Process quickly
- **Medium videos** (10-60 seconds): May take several seconds to process
- **Long videos** (> 60 seconds): Consider extracting at lower FPS or limiting frames

Processing time depends on:
- Video length and resolution
- Number of frames extracted
- Available system resources

### Memory Usage

Videos consume more memory than images due to:
- Multiple frames being processed
- Temporal patch creation
- 3D convolution operations

For long videos, consider:
- Reducing extraction FPS
- Setting a maximum frame limit
- Processing shorter clips

## Troubleshooting

### "ffmpeg is not installed or not in PATH"

**Problem**: Video processing requires FFmpeg but it's not found.

**Solution**: Install FFmpeg for your operating system:
```bash
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y ffmpeg

# macOS
brew install ffmpeg

# Windows
choco install -y ffmpeg
```

### "no frames extracted from video"

**Problem**: FFmpeg couldn't extract frames from the video.

**Possible causes**:
- Corrupted video file
- Unsupported codec
- Invalid video format

**Solution**: 
- Verify the video plays in a media player
- Try converting to MP4 with H.264 codec
- Check FFmpeg can process it: `ffmpeg -i video.mp4`

### Model responds with "I can't watch videos"

**Problem**: The model isn't recognizing video input.

**Possible causes**:
- Wrong model (not all models support videos)
- Video not properly encoded
- Server running old version without video support

**Solution**:
- Use a video-capable model like `qwen3-vl`
- Verify video is base64-encoded correctly
- Check Ollama version includes video support

## Examples

### Python Example

```python
import requests
import base64

# Read and encode video
with open('video.mp4', 'rb') as f:
    video_b64 = base64.b64encode(f.read()).decode()

# Send request
response = requests.post('http://localhost:11434/api/generate',
    json={
        "model": "qwen3-vl",
        "prompt": "Describe the actions in this video.",
        "videos": [video_b64],
        "stream": False
    }
)

print(response.json()['response'])
```

### JavaScript Example

```javascript
const fs = require('fs');
const axios = require('axios');

// Read and encode video
const videoBuffer = fs.readFileSync('video.mp4');
const videoBase64 = videoBuffer.toString('base64');

// Send request
axios.post('http://localhost:11434/api/generate', {
  model: 'qwen3-vl',
  prompt: 'What is happening in this video?',
  videos: [videoBase64],
  stream: false
}).then(response => {
  console.log(response.data.response);
});
```

### cURL with URL

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-vl",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this video"},
        {"type": "video_url", "video_url": {"url": "https://example.com/video.mp4"}}
      ]
    }]
  }'
```

## Limitations

1. **Model Support**: Only models specifically trained for video understanding can process videos (e.g., Qwen3-VL, Qwen2.5-VL)
2. **Frame Extraction**: Requires FFmpeg installation
3. **Processing Time**: Long videos take more time to process
4. **Memory**: Videos consume more memory than images
5. **Temporal Understanding**: Quality depends on model's temporal reasoning capabilities

## Best Practices

1. **Choose the right model**: Some models support reasoning and longer video context windows
2. **Use MP4 format**: Most compatible and efficient
3. **Keep videos short**: < 30 seconds for best performance (Qwen3-VL can handle longer videos)
4. **Appropriate FPS**: 1 FPS is usually sufficient for understanding
5. **Clear questions**: Ask specific questions about the video content
6. **Test locally first**: Verify videos work before deploying
7. **Monitor resources**: Watch memory usage with long videos

## Future Enhancements

Potential future improvements:
- Configurable FPS via API
- Video preprocessing (trimming, resizing)
- Streaming video support
- Multiple video inputs
- Audio processing support

## Contributing

To contribute video support improvements:
1. See `model/imageproc/video.go` for shared utilities
2. Model-specific processing in `model/models/qwen3vl/` and `model/models/qwen25vl/`
3. Tests in `model/imageproc/video_test.go`
4. Documentation updates welcome!

## Learn More

- [Qwen3-VL Model Card](https://ollama.com/library/qwen3-vl)
- [Qwen2.5-VL Model Card](https://ollama.com/library/qwen2.5-vl)
- [Ollama API Documentation](./api.md)
- [Multimodal Models](https://ollama.com/search?c=vision)
