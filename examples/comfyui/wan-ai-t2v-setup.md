# Wan-AI Wan2.1-T2V-14B Model Setup for ComfyUI

This guide provides instructions for setting up the Wan-AI Wan2.1-T2V-14B text-to-video model with ComfyUI. This is a 14 billion parameter model optimized for text-to-video generation.

## Overview

The Wan2.1-T2V-14B model is a powerful text-to-video generation model that requires several components to be downloaded and placed in specific directories within your ComfyUI installation.

## Prerequisites

1. **ComfyUI** installed and running
   - Clone from [GitHub](https://github.com/comfyanonymous/ComfyUI)
   - Follow the [installation guide](https://github.com/comfyanonymous/ComfyUI#installing)

2. **Sufficient Storage Space**
   - At least 40-50 GB of free disk space
   - Models are large and require significant storage

3. **GPU Requirements**
   - Minimum 16 GB VRAM recommended
   - 24 GB or more VRAM for optimal performance
   - FP8 quantized models are used to reduce memory requirements

## Required Files and Installation

### 1. Core Diffusion Models

These are the main model files for text-to-video generation. Both files are required.

**Files:**
- `wan2.1_t2v_14B_fp8_high_noise.safetensors` (approximately 14-15 GB)
- `wan2.1_t2v_14B_fp8_low_noise.safetensors` (approximately 14-15 GB)

**Installation Location:**
```
ComfyUI/models/diffusion_models/
```

**Alternative Location** (for older ComfyUI versions):
```
ComfyUI/models/unet/
```

**Note:** If the `diffusion_models` folder doesn't exist, it's recommended to create it:

```bash
mkdir -p ComfyUI/models/diffusion_models
```

### 2. VAE (Variational Autoencoder)

The VAE is responsible for decoding the latent representations into video frames.

**File:**
- `wan_2.1_vae.safetensors`

**Installation Location:**
```
ComfyUI/models/vae/
```

### 3. Text Encoder (CLIP/T5)

The text encoder processes your text prompts into embeddings used by the model.

**File:**
- `umt5_xxl_fp8_e4m3fn_scaled.safetensors` (approximately 4-5 GB)

**Installation Location:**
```
ComfyUI/models/text_encoders/
```

**Alternative Location** (for older ComfyUI versions):
```
ComfyUI/models/clip/
```

## Installation Steps

### Step 1: Download the Model Files

1. Search for **Wan-AI/Wan2.1-T2V-14B** model files (look for Repack versions)
2. Download all four required files listed above
3. Verify the file sizes match the expected sizes

**Recommended Sources:**
- Official Wan-AI GitHub repository: https://github.com/Wan-AI (check releases and model repositories)
- HuggingFace model hub: Search for "Wan2.1-T2V-14B" or "Wan-AI"
- Community repacks: Check trusted model sharing platforms and forums

**Important:** Always download from trusted sources. Verify file integrity by:
- Checking file sizes match expected values
- Comparing checksums/hashes if provided by the source
- Reading community reviews and feedback about the source

### Step 2: Create Required Directories

If the directories don't exist, create them:

```bash
# Navigate to your ComfyUI installation directory
cd /path/to/ComfyUI

# Create required directories
mkdir -p models/diffusion_models
mkdir -p models/vae
mkdir -p models/text_encoders
```

### Step 3: Move Files to Correct Locations

Move each downloaded file to its respective directory:

```bash
# Move diffusion models
mv wan2.1_t2v_14B_fp8_high_noise.safetensors ComfyUI/models/diffusion_models/
mv wan2.1_t2v_14B_fp8_low_noise.safetensors ComfyUI/models/diffusion_models/

# Move VAE
mv wan_2.1_vae.safetensors ComfyUI/models/vae/

# Move text encoder
mv umt5_xxl_fp8_e4m3fn_scaled.safetensors ComfyUI/models/text_encoders/
```

### Step 4: Refresh ComfyUI

After placing all files:

1. Restart ComfyUI if it's currently running
2. Launch ComfyUI: `python main.py` (or your preferred launch method)
3. Refresh the browser page to ensure models are loaded
4. Check the model selection dropdowns to verify the models appear

## Directory Structure

After installation, your ComfyUI models directory should look like this:

```
ComfyUI/
├── models/
│   ├── diffusion_models/
│   │   ├── wan2.1_t2v_14B_fp8_high_noise.safetensors
│   │   └── wan2.1_t2v_14B_fp8_low_noise.safetensors
│   ├── vae/
│   │   └── wan_2.1_vae.safetensors
│   └── text_encoders/
│       └── umt5_xxl_fp8_e4m3fn_scaled.safetensors
```

## Using the Model

### Basic Workflow

1. **Load the Model**: Select the Wan2.1 diffusion model in your workflow
2. **Load VAE**: Select the `wan_2.1_vae.safetensors` for the VAE node
3. **Load Text Encoder**: Select the `umt5_xxl_fp8_e4m3fn_scaled.safetensors` for text encoding
4. **Enter Prompt**: Provide your text description for the video
5. **Configure Parameters**: Set video length, resolution, and generation parameters
6. **Generate**: Queue the prompt to generate your video

### Recommended Settings

**For Best Quality:**
- Use both high_noise and low_noise models in appropriate workflow stages
- Higher resolution settings (limited by VRAM)
- More sampling steps (20-30 steps recommended)
- CFG scale: 7-9 for balanced guidance

**For Faster Generation:**
- Lower resolution settings
- Fewer sampling steps (15-20 steps)
- CFG scale: 5-7
- Consider using only one noise model if VRAM is limited

## Troubleshooting

### Model Not Appearing in ComfyUI

1. **Verify file locations**: Ensure files are in the correct directories
2. **Check file names**: Model file names must match exactly (case-sensitive)
3. **Restart ComfyUI**: Completely restart the ComfyUI application
4. **Clear cache**: Delete ComfyUI cache if models still don't appear
5. **Check console**: Look for error messages in the ComfyUI console output

### Out of Memory Errors

1. **Reduce resolution**: Lower the output video resolution
2. **Reduce batch size**: Generate one video at a time
3. **Close other applications**: Free up VRAM and system RAM
4. **Use FP8 models**: Ensure you're using the FP8 versions (these are already FP8)
5. **Upgrade GPU**: Consider a GPU with more VRAM for larger generations

### Slow Generation

1. **GPU acceleration**: Verify GPU is being used (check task manager/nvidia-smi)
2. **Optimize settings**: Reduce steps, resolution, or video length
3. **Update drivers**: Ensure GPU drivers are up to date
4. **Check temperatures**: Ensure GPU isn't thermal throttling
5. **System resources**: Close unnecessary background applications

### Corrupted or Incomplete Downloads

1. **Verify checksums**: Check MD5/SHA256 hashes if provided by your download source
2. **Compare file sizes**: Ensure downloaded files match expected sizes:
   - Core models: ~14-15 GB each
   - Text encoder: ~4-5 GB
   - VAE: Check size from source documentation
3. **Re-download**: Download files again if they seem corrupted
4. **Check disk space**: Ensure sufficient space during download (at least 50 GB free recommended)
5. **Use reliable sources**: Download from trusted, verified sources only

**Note:** Checksum values vary by source and version. Always obtain checksums from the same trusted source where you downloaded the files.

## Performance Considerations

### VRAM Requirements by Configuration

| Resolution | VRAM Required | Notes |
|-----------|---------------|-------|
| 512x512   | 12-16 GB      | Minimum viable |
| 768x768   | 16-20 GB      | Recommended minimum |
| 1024x1024 | 20-24 GB      | Good quality |
| 1280x1280 | 24+ GB        | High quality |

### Generation Time Estimates

Generation times vary based on:
- Video length (frames)
- Resolution
- Sampling steps
- GPU model and VRAM
- System configuration

**Example Times** (RTX 4090, 24GB VRAM):
- 3 second video (24 fps) at 768x768: 2-3 minutes
- 5 second video at 1024x1024: 4-6 minutes

## Model Information

### Architecture
- **Model Type**: Diffusion-based text-to-video
- **Parameters**: 14 billion
- **Precision**: FP8 quantized for efficiency
- **Context**: Two-stage noise model (high_noise and low_noise)

### Capabilities
- Text-to-video generation
- Multiple aspect ratios support
- Adjustable video length
- High-quality output with proper settings

### Limitations
- Requires significant VRAM (16+ GB recommended)
- Long generation times for high-quality videos
- May struggle with complex motion or scene changes
- Limited to trained video length/resolution ranges

## Additional Resources

- [Wan-AI Official Repository](https://github.com/Wan-AI/Wan2.1-T2V-14B)
- [ComfyUI Documentation](https://github.com/comfyanonymous/ComfyUI)
- [ComfyUI Workflows](./README.md)
- [Model Fine-tuning Guide](https://github.com/Wan-AI/Wan2.1-T2V-14B/blob/main/docs/fine-tuning.md)

## Community and Support

For questions, issues, or sharing results:
- ComfyUI Discord
- Wan-AI GitHub Discussions
- Reddit r/StableDiffusion and r/ComfyUI
- HuggingFace model page comments

## License and Terms

Please review the Wan-AI/Wan2.1-T2V-14B model license before use:
- Commercial use restrictions may apply
- Attribution requirements
- Ethical use guidelines
- Distribution terms

Always ensure you comply with the model's license and terms of use.

## Updates and Versions

This guide is for Wan2.1-T2V-14B. Check for newer versions:
- Model updates and improvements
- Bug fixes and optimizations
- New features and capabilities

Keep your model files updated for the best experience.

## FAQ

**Q: Do I need all four files?**
A: Yes, all four files are required for the model to function properly.

**Q: Can I use only one noise model file?**
A: While possible in some workflows, using both provides better quality results across different generation scenarios.

**Q: What if I have an older ComfyUI version?**
A: Use the alternative folder locations (unet/ and clip/) if diffusion_models/ or text_encoders/ don't exist.

**Q: Can I run this on a GPU with less than 16 GB VRAM?**
A: It may work with lower resolutions and optimized settings, but 16+ GB is strongly recommended.

**Q: Where can I find example workflows?**
A: Check the ComfyUI community, model repository, or create custom workflows based on standard text-to-video nodes.

**Q: How do I update the model?**
A: Download new versions of the model files and replace the old ones in the same directories, then restart ComfyUI.

## Contributing

If you discover improvements or have additional tips for using this model:
1. Test thoroughly
2. Document your findings
3. Share with the community
4. Consider contributing to this guide

---

For more ComfyUI workflows and examples, see the [main ComfyUI README](./README.md).
