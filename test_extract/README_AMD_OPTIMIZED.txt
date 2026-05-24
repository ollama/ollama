========================================================================
Ollama for AMD & Windows - Custom Build (RDNA4 / GFX1201 Optimized)
========================================================================
This is a custom-compiled version of Ollama with full acceleration for:
- AMD GPU Hardware Acceleration (ROCm 7.1 / gfx1201)
- Vulkan Graphics API Acceleration
- High-Performance Vector CPU Backends (Haswell, Skylake-X, Alder Lake, etc.)

Instructions:
1. Extract all files (including the 'lib' folder) to your chosen directory.
2. Open PowerShell in that directory.
3. Start the server:
   .\ollama.exe serve
4. Run inference:
   .\ollama.exe run llama3

Performance features integrated:
- 64-bit Warp Mask (the 65% performance hack)
- rocWMMA Flash Attention kernels enabled
========================================================================
