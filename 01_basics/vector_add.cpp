#include <Kokkos_Core.hpp>
#include <cstdio>

// Masalah: Penjumlahan Vektor C = A + B
// Tujuan: Memahami View, Mirror, dan DeepCopy

int main(int argc, char* argv[]) {
  // 1. Inisialisasi Kokkos (Wajib!)
  Kokkos::initialize(argc, argv);
  {
    const int N = 1000;
    printf("Menghitung Vector Add dengan N=%d pada Execution Space Default...\n", N);

    // 2. Alokasi Memori di DEVICE (GPU jika pakai CUDA, CPU jika pakai OpenMP/Serial)
    // View adalah array multidimensi pengelola memori otomatis.
    Kokkos::View<double*> d_A("dev_A", N);
    Kokkos::View<double*> d_B("dev_B", N);
    Kokkos::View<double*> d_C("dev_C", N);

    // 3. Alokasi Mirror di HOST (CPU RAM) untuk inisialisasi data
    // create_mirror_view akan membuat View di CPU yang kompatibel dengan Device.
    auto h_A = Kokkos::create_mirror_view(d_A);
    auto h_B = Kokkos::create_mirror_view(d_B);
    auto h_C = Kokkos::create_mirror_view(d_C);

    // 4. Inisialisasi Data di HOST (Standard C++ loop)
    for (int i = 0; i < N; ++i) {
        h_A(i) = 1.0 * i;
        h_B(i) = 2.0 * i;
    }

    // 5. Deep Copy: Host -> Device
    // Ini menggantikan cudaMemcpy yang ribet.
    Kokkos::deep_copy(d_A, h_A);
    Kokkos::deep_copy(d_B, h_B);

    // 6. Parallel Compute (Kernel)
    // KOKKOS_LAMBDA menggantikan __device__ function di CUDA.
    // parallel_for menggantikan loop for(int i=0; i<N; i++)
    Kokkos::parallel_for("VectorAdd", N, KOKKOS_LAMBDA(const int i) {
        d_C(i) = d_A(i) + d_B(i);
    });
    
    // Tunggu GPU selesai (jika async)
    Kokkos::fence();

    // 7. Deep Copy Balik: Device -> Host
    Kokkos::deep_copy(h_C, d_C);

    // 8. Verifikasi Hasil di Host
    printf("Hasil d_C(5) = %f (Seharusnya 15.0)\n", h_C(5));
  }
  // 9. Finalisasi
  Kokkos::finalize();
  return 0;
}
