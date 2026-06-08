# Week 1 Findings - Internship Experience

## Overview
This document summarizes my experiences and learnings during the first week of my internship, focusing on understanding IBM's software infrastructure and working on the LLM inference project.

## Key Learnings

### IBM Software Infrastructure
During the first week, I gained valuable exposure to IBM's software infrastructure. This included understanding the architecture, development workflows, and the tools used across the organization. The infrastructure's complexity and scale provided important context for the work ahead.

### Lower Level Concepts and Their Relevance
I spent considerable time deepening my understanding of lower-level computing concepts, particularly:

- **Compilers**: How source code is translated into executable machine code, including optimization techniques and target architecture considerations
- **Operating Systems**: Core OS concepts including process management, memory allocation, and system calls

These concepts proved directly relevant to the LLM inference project, as understanding how code is compiled and executed at the system level is crucial for optimizing model performance and troubleshooting platform-specific issues.

## Technical Challenges

### Mainframe Access Through Containers
One significant challenge this week was the reduced development velocity due to accessing the mainframe through a container environment. This added layer of abstraction introduced latency and complexity to the development workflow, requiring adjustments to my typical development practices.

### Container Technology
As part of addressing the mainframe access challenges, I invested time in learning about container technology, including:
- Container fundamentals and architecture
- How containers provide isolation and portability
- Best practices for working within containerized environments

This knowledge will be valuable for future development work and understanding deployment strategies.

## Build Attempts

### Ollama Repository - Partial Success
I successfully built the ollama repository, which was a significant milestone. The build process completed without errors, demonstrating that the development environment was properly configured for compilation.

**However**, after the successful build, I encountered issues when attempting to run models. The built binaries were unable to execute any models, indicating potential runtime configuration issues or missing dependencies that weren't caught during the build phase.

### Llama.cpp Repository - Build Failure
I also attempted to build the llama.cpp repository, which unfortunately failed. This failure requires further investigation to determine whether it's related to:
- Platform-specific compatibility issues
- Missing dependencies
- Configuration problems in the containerized environment
- Architecture-specific build requirements

## Next Steps
- Investigate the runtime issues preventing model execution in the ollama build
- Debug and resolve the llama.cpp build failures
- Continue deepening understanding of the relationship between system-level concepts and LLM inference optimization
- Improve efficiency when working within the containerized mainframe environment

## Reflections
Despite the technical challenges encountered this week, the experience has been valuable for building foundational knowledge. Understanding the lower-level concepts and their practical application to LLM inference will be crucial for the success of this project. The build issues, while frustrating, provide important learning opportunities for troubleshooting and problem-solving in complex development environments.