# cuQuant8

A high-performance, custom CUDA implementation of Symmetric (Absmax) Quantization and INT8 Matrix Multiplication, leveraging NVIDIA Tensor Cores via `cuBLASLt`. Exposed as a drop-in PyTorch C++ Extension.

![CUDA Version](https://img.shields.io/badge/CUDA-11.8%2B-green) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red) ![Hardware](https://img.shields.io/badge/Hardware-Ampere%20%7C%20Hopper-blue)

##  Performance Highlights
*If you only read one section, read this.*

* **Memory Reduction:** Achieved a **4x reduction** in weight memory footprint (FP32 -> INT8) with `< 0.1%` degradation in accuracy on standard evaluation distributions.
* **Throughput Gain:** `cuBLASLt` INT8 MatMul achieved a **[X.X]x speedup** over PyTorch's native FP32 `torch.matmul` on an RTX2060.
* **Bandwidth Utilization:** Custom parallel reduction kernel for finding `absmax` utilizes **[X]% of peak memory bandwidth** (profiled via Nsight Compute).

*(Reminder to add a clean bar chart here comparing FP32 vs INT8 execution time across different batch sizes)*

## Core Features
* **Custom Absmax Kernel:** Hand-written CUDA kernel using warp-level primitives (`__shfl_down_sync`) for highly optimized parallel reductions to find the absolute maximum for scaling.
* **Outlier Clipping:** Configurable clamp threshold to preserve core data resolution and prevent extreme outliers from crushing the INT8 distribution.
* **Tensor Core Integration:** Bypasses standard CUDA cores, utilizing `cublasLtMatmul` to unlock hardware-accelerated mixed-precision math (`CUDA_R_8I` inputs to `CUDA_R_32F` output).
* **PyTorch C++ Backend:** Seamlessly wrapped using `pybind11` and `torch.utils.cpp_extension` to allow direct integration into standard PyTorch `nn.Module` layers.

## Hardware & Software Requirements
* **GPU:** NVIDIA GPU with Compute Capability 8.0+ (Ampere, Ada, Hopper) required for specific Tensor Core instructions.
* **OS:** Linux (Ubuntu 20.04/22.04) or WSL2.
* **CUDA:** Toolkit 11.8 or newer.
* **Compiler:** `nvcc` and `g++` (C++17).

## Build & Installation

```bash
git clone https://github.com/brandonviaje/cuQuant8.git
cd cuQuant8

# Install via PyTorch setup.py
pip install .

# Or build the raw C++ tests via CMake
mkdir build && cd build
cmake ..
make
./quant_test
```
