# Kokkos SpMV Experiment: Reordering Effects 🚀

This repository contains my personal implementation of Sparse Matrix-Vector Multiplication (SpMV) using **Kokkos Framework** (C++ Performance Portability). The goal is to study the impact of **Graph Reordering (METIS)** on SpMV performance, inspired by the work of *Islam et al. (SC '25)*.

## 📂 Project Structure
*   `01_basics`: Introduction to Kokkos Views & Parallel Dispatch.
*   `02_memory`: Understanding Parallel Reduction & Memory Spaces.
*   `03_capstone`: Baseline SpMV Kernel Implementation (CSR Format).
*   `05_reordering`: Advanced experiment integrating **METIS NodeND** to reorder random/stencil matrices for cache locality optimization.

## 📊 Experimental Results (Preliminary)
I conducted a benchmark on a standard workstation (CPU OpenMP Backend) using a **Shuffled 3D 7-Point Stencil** matrix.

| Matrix Size | Method | Time (s) | GFLOPs | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **512k (80^3)** | Baseline | 0.051 | 0.12 | - |
| | **METIS** | 0.066 | 0.09 | 0.77x |
| **3.3M (150^3)** | Baseline | 0.098 | 0.41 | - |
| | **METIS** | 0.099 | 0.40 | **0.99x** |

### 💡 Analysis
Contrary to initial expectations, METIS reordering did not yield significant speedup on my CPU environment for 3D Stencil grids.
*   **Hypothesis:** The hardware prefetcher on modern CPUs handles the semi-structured randomness of shuffled grids effectively.
*   **Bottleneck:** The performance drop at 3.3M size suggests the workload is **Memory Bandwidth Bound** rather than Latency Bound, diminishing the benefits of cache locality optimization.

## 🛠️ How to Build
Requirements: `CMake`, `Kokkos`, `OpenMP`, `libmetis-dev`.

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_OPENMP=ON
make
./05_reordering
```