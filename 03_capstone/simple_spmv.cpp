#include <Kokkos_Core.hpp>
#include <cstdio>
#include <vector>

// MODUL 3: Simple SpMV (Sparse Matrix-Vector Multiplication) y = A * x
// Format: CSR (Compressed Sparse Row)

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    // --- 1. SETUP DATA (HOST) ---
    // Kita buat matriks 4x4 sederhana di CPU dulu:
    // A = [ 10  0  0 20 ]
    //     [  0 30  0  0 ]
    //     [ 40  0 50 60 ]
    //     [  0  0  0 70 ]
    // x = [ 1, 1, 1, 1 ]^T
    // Hasil y seharusnya = [30, 30, 150, 70]^T

    const int num_rows = 4;
    const int num_nnz = 7; // Jumlah non-zero elements

    // CSR Format Arrays (di Host/CPU biasa std::vector)
    std::vector<int> h_row_map = {0, 2, 3, 6, 7}; // Penanda awal tiap baris. Size = rows + 1
    std::vector<int> h_col_idx = {0, 3, 1, 0, 2, 3, 3}; // Kolom indikator untuk tiap nilai
    std::vector<double> h_values = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0};
    
    // --- 2. MOVE TO DEVICE (KOKKOS) ---
    
    // Alokasi View di Device
    Kokkos::View<int*>      row_map("row_map", num_rows + 1);
    Kokkos::View<int*>      col_idx("col_idx", num_nnz);
    Kokkos::View<double*>   values("values", num_nnz);
    Kokkos::View<double*>   x("x", num_rows);
    Kokkos::View<double*>   y("y", num_rows); // Hasil

    // Copy data dari std::vector ke Kokkos::View
    // Kita perlu "HostMirror" untuk jembatan copy
    auto h_row_map_view = Kokkos::create_mirror_view(row_map);
    auto h_col_idx_view = Kokkos::create_mirror_view(col_idx);
    auto h_values_view  = Kokkos::create_mirror_view(values);
    auto h_x_view       = Kokkos::create_mirror_view(x);

    // Isi mirror view
    for(int i=0; i<h_row_map.size(); i++) h_row_map_view(i) = h_row_map[i];
    for(int i=0; i<h_col_idx.size(); i++) h_col_idx_view(i) = h_col_idx[i];
    for(int i=0; i<h_values.size(); i++)  h_values_view(i)  = h_values[i];
    for(int i=0; i<num_rows; i++)         h_x_view(i)       = 1.0; // Vector x semua 1

    // Deep Copy ke Device
    Kokkos::deep_copy(row_map, h_row_map_view);
    Kokkos::deep_copy(col_idx, h_col_idx_view);
    Kokkos::deep_copy(values, h_values_view);
    Kokkos::deep_copy(x, h_x_view);

    printf("Menghitung SpMV pada Device...\n");

    // --- 3. PARALLEL KERNEL (THE CORE) ---
    
    // Logic SpMV:
    // Untuk setiap baris 'i':
    //    sum = 0
    //    Loop dari row_start sampai row_end:
    //       ambil value dan col_idx
    //       sum += value * x[col_idx]
    //    y[i] = sum

    // Kita pakai 'TeamPolicy' nanti untuk advanced, tapi sekarang pakai 'RangePolicy' (1 thread = 1 row)
    // yang paling sederhana dulu.

    Kokkos::parallel_for("SpMV_Kernel", num_rows, KOKKOS_LAMBDA(const int i) {
        double sum = 0.0;
        int row_start = row_map(i);
        int row_end   = row_map(i+1);

        for (int k = row_start; k < row_end; k++) {
            // k adalah index pointer ke array values/col_idx linear
            int col = col_idx(k);
            double val = values(k);
            
            sum += val * x(col);
        }
        
        y(i) = sum;
    });
    
    Kokkos::fence(); // Tunggu GPU

    // --- 4. VERIFIKASI ---
    auto h_y = Kokkos::create_mirror_view(y);
    Kokkos::deep_copy(h_y, y);

    printf("Hasil Y: [ ");
    for(int i=0; i<num_rows; i++) printf("%.1f ", h_y(i));
    printf("]\n");
    printf("Ekspektasi: [ 30.0 30.0 150.0 70.0 ]\n");
  }
  Kokkos::finalize();
  return 0;
}
