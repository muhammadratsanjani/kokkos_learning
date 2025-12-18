cmake_minimum_required(VERSION 3.16)
project(KokkosLearningPlan CXX)

# --- PERFORMANCE FLAGS (CRITICAL) ---
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native") # Paksa optimasi maksimal & instruksi CPU lokal


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

# --- MODULE 4: BENCHMARK SPMV (SCALING UP) ---
add_executable(04_benchmark 04_benchmark/benchmark_spmv.cpp)
target_link_libraries(04_benchmark Kokkos::kokkos)

# --- MODULE 5: REORDERING EXPERIMENT (METIS) ---
find_library(METIS_LIB metis)
if(NOT METIS_LIB)
    message(WARNING "Metis library not found! Module 5 might fail to link. Install with: sudo apt install libmetis-dev")
endif()

add_executable(05_reordering 05_reordering/spmv_metis.cpp)
target_link_libraries(05_reordering Kokkos::kokkos ${METIS_LIB})

# --- MODULE 6: GPU PREPARATION (TeamPolicy) ---
add_executable(06_gpu_ready 06_gpu_preparation/spmv_gpu.cpp)
target_link_libraries(06_gpu_ready Kokkos::kokkos)
