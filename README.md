# Kokkos SpMV Experiment: Reordering Effects

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/muhammadratsanjani/spmm-reordering-study/blob/main/Reproducibility_Colab.ipynb)

This repository contains my personal implementation of Sparse Matrix-Vector Multiplication (SpMV) using **Kokkos Framework** (C++ Performance Portability). The goal is to study the impact of **Graph Reordering (METIS)** on SpMV performance, inspired by the work of *Islam et al. (SC '25)*.

## 📂 Project Structure
*   `01_basics`: Introduction to Kokkos Views & Parallel Dispatch.
*   `02_memory`: Understanding Parallel Reduction & Memory Spaces.
*   `03_capstone`: Baseline SpMV Kernel Implementation (CSR Format).
*   `05_reordering`: Advanced experiment integrating **METIS NodeND** to reorder random/stencil matrices for cache locality optimization.
*   `06_gpu_preparation`: Hierarchical Parallelism (`TeamPolicy`) implementation ready for Cuda/HIP backends.
*   `07_gpu_benchmark`: Large-scale 3D Stencil generator for GPU performance validation.

## 📊 Experimental Results (Preliminary)
I conducted a benchmark on a standard workstation (CPU OpenMP Backend) and NVIDIA Tesla T4 (GPU Cuda Backend) using a **Shuffled 3D 7-Point Stencil** matrix.

| Matrix Size | Hardware | Method | Time (s) | GFLOPs | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **512k (80^3)** | CPU (Intel Core i7-1165G7) | OpenMP | 0.051 | **2.53** | Cache-friendly |
| | GPU (Tesla T4) | Naive Cuda | 0.015 | **0.46** | Memory Latency Bound! |
| **3.3M (150^3)** | CPU | OpenMP | 0.098 | 0.41 | Bandwidth Bound |

### 💡 Analysis
Our GPU benchmark (via Google Colab) revealed a **5x performance drop** compared to CPU when handling unstructured (shuffled) grids without reordering.
*   **Hypothesis:** The hardware prefetcher on modern CPUs handles the semi-structured randomness of shuffled grids effectively, whereas GPUs suffer from **uncoalesced memory access penalties** and warp divergence.
*   **Conclusion:** Graph Reordering (like METIS or RCM) is not optional but **critical** for GPU-based SpMV to unlock the hardware's potential (theoretical peak > 100 GFLOPs).

## 🛠️ How to Build
Requirements: `CMake`, `Kokkos`, `OpenMP`, `libmetis-dev`.

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_OPENMP=ON
make
./05_reordering
```

## Future Work
Porting to GPU using Kokkos Cuda Backend

## Reproducibility
Reproducibility_Colab.ipynb
