# Pull Request: Vulkan-Order GPU Selection (Commit 1f279e404e96deda6d4922a67c25cc642f8ca3be)

## Title
Fix: GPU Selection Respects Vulkan Order, Intel Overflow Only if Needed

## Description
This PR implements a GPU selection mode where the system strictly respects Vulkan device enumeration order (ID). The main GPU (ID=0, typically NVIDIA) is always used first; overflow to Intel (ID=1) only occurs if the model size exceeds the main GPU's VRAM.

## Motivation and Context
- Ensures hardware preference and system configuration are respected.
- Prevents less powerful GPUs from being selected solely due to higher VRAM.
- Provides clear and predictable GPU usage for advanced users.

## How Has This Been Tested?
- Loaded multiple models and verified layer assignment and overflow via logs.
- Example log: `GPULayers:41[ID:0 Layers:16(0..15) ID:1 Layers:25(16..40)]` (main GPU first, overflow on secondary).
- Confirmed that Intel GPU is only used when overflow is required.

## Types of Changes
- [x] New feature (non-breaking change)
- [x] Bug fix
- [ ] Documentation update

## Checklist
- [x] My code follows the code style of this project
- [x] I have added tests/logs to confirm correct behavior
- [x] I have updated documentation where necessary

## Reference
This PR is based on commit [1f279e404e96deda6d4922a67c25cc642f8ca3be](https://github.com/iosub/ollama/commit/1f279e404e96deda6d4922a67c25cc642f8ca3be).

---

**Author:** GitHub Copilot
**Date:** 2025-10-20
