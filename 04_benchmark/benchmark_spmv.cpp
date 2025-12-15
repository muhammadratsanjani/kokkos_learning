#include <Kokkos_Core.hpp>
#include <Kokkos_Timer.hpp> // Wajib untuk benchmark
#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>

// MODUL 4: BENCHMARKING & SCALING UP
// Kita tidak bisa melihat performa kalau matriks cuma 4x4.
// Kita akan generate matriks acak ukuran besar (N = 100.000+)
// dan mengukur GFLOPs (Giga Floating Point Operations per Second).

struct CSRMatrix {
    std::vector<int> row_map;
    std::vector<int> col_idx;
    std::vector<double> values;
    int num_rows;
    int num_nnz;
};

// Generator Matriks Random Sederhana
CSRMatrix generate_random_csr(int rows, int cols, double density) {
    CSRMatrix mat;
    mat.num_rows = rows;
    mat.row_map.push_back(0);
    
    std::mt19937 rng(12345); // Seed tetap agar reproducible
    std::uniform_real_distribution<double> dist_val(0.0, 10.0);
    std::uniform_int_distribution<int> dist_col(0, cols - 1);

    int current_nnz = 0;
    for (int i = 0; i < rows; ++i) {
        // Tentukan jumlah nnz di baris ini (rata-rata berdasarkan density)
        // Agar simpel, kita buat fix 50-100 elemen per baris biar "berat"
        int row_nnz = 50 + (rng() % 50); 
        
        std::vector<int> col_indices;
        while(col_indices.size() < row_nnz) {
            int c = dist_col(rng);
            // Cek duplikat (inefisien tapi oke untuk init)
            bool duplicate = false;
            for(int existing : col_indices) if(existing == c) duplicate = true;
            if(!duplicate) col_indices.push_back(c);
        }
        std::sort(col_indices.begin(), col_indices.end()); // CSR wajib urut kolomnya

        for(int c : col_indices) {
            mat.col_idx.push_back(c);
            mat.values.push_back(dist_val(rng));
            current_nnz++;
        }
        mat.row_map.push_back(current_nnz);
    }
    mat.num_nnz = current_nnz;
    return mat;
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    int N = 100000; // 100 Ribu Baris
    printf("Generating Random Matrix %dx%d...\n", N, N);
    
    CSRMatrix h_mat = generate_random_csr(N, N, 0.01);
    printf("Matrix Generated. NNZ = %d\n", h_mat.num_nnz);

    // --- SETUP DEVICE VIEWS ---
    Kokkos::View<int*>      row_map("row_map", h_mat.num_rows + 1);
    Kokkos::View<int*>      col_idx("col_idx", h_mat.num_nnz);
    Kokkos::View<double*>   values("values", h_mat.num_nnz);
    Kokkos::View<double*>   x("x", N);
    Kokkos::View<double*>   y("y", N);

    // Copy to Device
    auto h_row_map_v = Kokkos::create_mirror_view(row_map);
    auto h_col_idx_v = Kokkos::create_mirror_view(col_idx);
    auto h_values_v  = Kokkos::create_mirror_view(values);
    auto h_x_v       = Kokkos::create_mirror_view(x);

    // Salin data vector ke view
    for(size_t i=0; i<h_mat.row_map.size(); i++) h_row_map_v(i) = h_mat.row_map[i];
    for(size_t i=0; i<h_mat.col_idx.size(); i++) h_col_idx_v(i) = h_mat.col_idx[i];
    for(size_t i=0; i<h_mat.values.size(); i++)  h_values_v(i)  = h_mat.values[i];
    for(int i=0; i<N; i++) h_x_v(i) = 1.0;

    Kokkos::deep_copy(row_map, h_row_map_v);
    Kokkos::deep_copy(col_idx, h_col_idx_v);
    Kokkos::deep_copy(values, h_values_v);
    Kokkos::deep_copy(x, h_x_v);

    // --- WARMUP ---
    // Jalankan sekali agar cache/GPU "panas" (menghindari overhead inisialisasi awal)
    Kokkos::parallel_for("SpMV_Warmup", N, KOKKOS_LAMBDA(const int i) {
        double sum = 0.0;
        int start = row_map(i);
        int end   = row_map(i+1);
        for (int k = start; k < end; k++) {
            sum += values(k) * x(col_idx(k));
        }
        y(i) = sum;
    });
    Kokkos::fence();

    // --- TIMING LOOP ---
    Kokkos::Timer timer; // Timer mulai hitung otomatis
    const int REPEAT = 100;
    
    for(int iter=0; iter<REPEAT; iter++) {
        Kokkos::parallel_for("SpMV_Run", N, KOKKOS_LAMBDA(const int i) {
            double sum = 0.0;
            int start = row_map(i);
            int end   = row_map(i+1);
            for (int k = start; k < end; k++) {
                sum += values(k) * x(col_idx(k));
            }
            y(i) = sum;
        });
    }
    Kokkos::fence(); // Wajib tunggu sebelum ambil waktu
    double time_seconds = timer.seconds();

    // --- REPORT ---
    double avg_time = time_seconds / REPEAT;
    // GFLOPs = (2 * NNZ) / time (karena 1 elemen = 1 kali + 1 tambah)
    double gflops = (2.0 * h_mat.num_nnz * 1e-9) / avg_time;
    
    printf("Selesai %d Iterasi.\n", REPEAT);
    printf("Total Waktu: %f s\n", time_seconds);
    printf("Avg Waktu  : %f s\n", avg_time);
    printf("Performance: %f GFLOPs\n", gflops);
  }
  Kokkos::finalize();
  return 0;
}
