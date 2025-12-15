#include <Kokkos_Core.hpp>
#include <cstdio>

// MODUL 2: PARALLEL REDUCTION
// Masalah: Menghitung Dot Product (A . B) = sum(A[i] * B[i])
// Tantangan: Ribuan thread mencoba menulis ke variabel 'total' yang sama secara bersamaan (Race Condition).
// Solusi Tradisional: Atomic operations (lambat) atau Shared Memory reduction (rumit).
// Solusi Kokkos: parallel_reduce (Portable & Optimized).

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    const int N = 10000; 
    printf("Menghitung Dot Product dengan N=%d...\n", N);

    // 1. Alokasi Memori (Sama seperti sebelumnya)
    Kokkos::View<double*> d_A("dev_A", N);
    Kokkos::View<double*> d_B("dev_B", N);

    auto h_A = Kokkos::create_mirror_view(d_A);
    auto h_B = Kokkos::create_mirror_view(d_B);

    // 2. Inisialisasi Data
    // Kita isi A dengan 1.0 dan B dengan 2.0
    // Hasil ekspektasi: 1.0 * 2.0 * 10000 = 20000.0
    for (int i = 0; i < N; ++i) {
        h_A(i) = 1.0;
        h_B(i) = 2.0; 
    }

    Kokkos::deep_copy(d_A, h_A);
    Kokkos::deep_copy(d_B, h_B);

    // 3. Parallel Reduce
    double final_sum = 0.0; // Variabel di Host untuk menyimpan hasil

    // Perhatikan Lambda Signaturnya:
    // (const int i, double& lsum)
    // - i    : Index loop (seperti biasa)
    // - lsum : "Local Sum" -> Variabel sementara milik thread/block.
    //          Kokkos akan otomatis menggabungkan semua 'lsum' dari semua thread
    //          menjadi 'final_sum' di akhir eksekusi.
    
    Kokkos::parallel_reduce("DotProduct kernel", N, KOKKOS_LAMBDA(const int i, double& lsum) {
        lsum += d_A(i) * d_B(i); // Akumulasi ke local sum, BUKAN ke global sum
    }, final_sum);

    printf("Hasil Dot Product: %f (Seharusnya 20000.00)\n", final_sum);
  }
  Kokkos::finalize();
  return 0;
}
