# Pull Request: VRAM-Based GPU Selection (Commit d4bd02652234e1426b4dde540cd9ad48f57b983a)

## Title
Feature: GPU Selection by Available VRAM

## Description
This PR implements a GPU selection mode where the system prioritizes the GPU with the highest available memory (VRAM) for model loading. If the model size exceeds the main GPU's capacity, overflow is handled by offloading layers to additional GPUs.

## Motivation and Context
- Maximizes total VRAM usage for large models.
- Useful for heterogeneous systems with multiple GPUs of different capacities.
- Addresses user requests for memory-prioritized selection.

## How Has This Been Tested?
- Loaded multiple models and verified layer assignment via logs.
- Example log: `GPULayers:41[ID:1 Layers:41(0..40)]` (all layers on the GPU with most VRAM).

## Types of Changes
- [x] New feature (non-breaking change)
- [ ] Bug fix
- [ ] Documentation update

## Checklist
- [x] My code follows the code style of this project
- [x] I have added tests/logs to confirm correct behavior
- [x] I have updated documentation where necessary

## Reference
This PR is based on commit [d4bd02652234e1426b4dde540cd9ad48f57b983a](https://github.com/iosub/ollama/commit/d4bd02652234e1426b4dde540cd9ad48f57b983a).

---

**Author:** GitHub Copilot
**Date:** 2025-10-20
