#!/bin/bash
# Setup environment for MoDE_Diffusion_Policy

# Fix cuDNN compatibility issues by removing conflicting CUDA paths
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | sed 's|/usr/local/cuda[^:]*:||g' | sed 's|::|:|g' | sed 's|^:||' | sed 's|:$||')