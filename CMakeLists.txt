cmake_minimum_required(VERSION 3.16)
project(KokkosLearningPlan CXX)

# --- KONFIGURASI KOKKOS ---
# Pastikan Kokkos sudah terinstall atau ada di subdirectory.
# Cara paling mudah: git clone https://github.com/kokkos/kokkos.git di dalam folder ini
# Lalu gunakan: add_subdirectory(kokkos)

# Opsi Kokkos (Sesuaikan dengan Hardware Anda!)
# Jika pakai GPU NVIDIA: -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_VOLTA70=ON (sesuaikan arsitektur)
# Jika pakai CPU Multi-core: -DKokkos_ENABLE_OPENMP=ON

add_subdirectory(kokkos)

# --- MODULE 1: VECTOR ADD ---
add_executable(01_vector_add 01_basics/vector_add.cpp)
target_link_libraries(01_vector_add Kokkos::kokkos)

# --- MODULE 2: DOT PRODUCT (REDUCE) ---
add_executable(02_dot_product 02_memory/dot_product.cpp)
target_link_libraries(02_dot_product Kokkos::kokkos)

# --- MODULE 3: SIMPLE SPMV (CAPSTONE) ---
add_executable(03_spmv 03_capstone/simple_spmv.cpp)
target_link_libraries(03_spmv Kokkos::kokkos)
