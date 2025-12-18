#include <Kokkos_Core.hpp>
#include <cstdio>
#include <vector>

// MODUL 6: GPU-READY SPMV (HIERARCHICAL PARALLELISM)
// Tujuan: Menggunakan TeamPolicy untuk memanfaatkan struktur Grid/Block/Thread GPU.

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    // 1. SETUP DATA (Sama seperti Modul 3 agar mudah diverifikasi)
    const int num_rows = 4;
    const int num_nnz = 7; 

    // Host Data
    std::vector<int> h_row_map = {0, 2, 3, 6, 7};
    std::vector<int> h_col_idx = {0, 3, 1, 0, 2, 3, 3};
    std::vector<double> h_values = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0};
    
    // Device Views
    // --- GPU NOTE 1: Mem Space ---
    // Kokkos::DefaultExecutionSpace bisa CUDA (jika ada) atau OpenMP.
    // MemorySpace otomatis menyesuaikan (CudaSpace atau HostSpace).
    Kokkos::View<int*>      row_map("row_map", num_rows + 1);
    Kokkos::View<int*>      col_idx("col_idx", num_nnz);
    Kokkos::View<double*>   values("values", num_nnz);
    Kokkos::View<double*>   x("x", num_rows);
    Kokkos::View<double*>   y("y", num_rows);

    // Deep Copy
    auto h_row_map_v = Kokkos::create_mirror_view(row_map);
    auto h_col_idx_v = Kokkos::create_mirror_view(col_idx);
    auto h_values_v  = Kokkos::create_mirror_view(values);
    auto h_x_v       = Kokkos::create_mirror_view(x);

    for(int i=0; i<h_row_map.size(); i++) h_row_map_v(i) = h_row_map[i];
    for(int i=0; i<h_col_idx.size(); i++) h_col_idx_v(i) = h_col_idx[i];
    for(int i=0; i<h_values.size(); i++)  h_values_v(i)  = h_values[i];
    for(int i=0; i<num_rows; i++)         h_x_v(i)       = 1.0;

    Kokkos::deep_copy(row_map, h_row_map_v);
    Kokkos::deep_copy(col_idx, h_col_idx_v);
    Kokkos::deep_copy(values, h_values_v);
    Kokkos::deep_copy(x, h_x_v);

    printf("Menghitung GPU-Ready SpMV (TeamPolicy)...\n");

    // --- GPU NOTE 2: HIERARCHICAL PARALLELISM ---
    
    // TeamPolicy<ExecutionSpace>(LeagueSize, TeamSize)
    // - LeagueSize: Jumlah "Tim" (mirip CUDA Grid Size / Blocks). Biasanya = Jumlah Baris.
    // - TeamSize: Jumlah Thread per Tim (mirip CUDA Block Size / Threads per Block). 
    //             Kokkos bisa pilih otomatis dengan Kokkos::AUTO.
    
    typedef Kokkos::TeamPolicy<> policy_t;
    typedef policy_t::member_type member_t; // Member adalah satu thread dalam tim

    Kokkos::parallel_for("SpMV_Team", policy_t(num_rows, Kokkos::AUTO), KOKKOS_LAMBDA(const member_t& team_member) {
        
        // 1. Dapatkan Baris yang dikerjakan oleh Tim ini
        // team_member.league_rank() = Block ID (0..num_rows-1)
        int row = team_member.league_rank(); 

        double row_sum = 0.0;
        int row_start = row_map(row);
        int row_end   = row_map(row+1);
        int row_len   = row_end - row_start;

        // 2. Parallel Reduce DALAM TIM (Intra-Team)
        // Semua thread dalam satu tim bekerja sama menghitung satu baris.
        // Ini disebut "Thread-Level Parallelism" untuk menangani baris yang sangat panjang (mencegah load imbalance).
        // Kokkos::TeamThreadRange(team_member, count) -> Loop paralel selebar tim.
        
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team_member, row_len), [=](const int& k_offset, double& lsum) {
            // k_offset adalah 0, 1, 2... relatif terhadap awal loop
            int k = row_start + k_offset; 
            lsum += values(k) * x(col_idx(k));
        }, row_sum);

        // 3. Single Thread Write
        // Hanya satu threads yg menulis hasil akhir ke memori global
        team_member.team_barrier(); // Tunggu semua hitung selesai (barrier)
        if (team_member.team_rank() == 0) { // Thread 0 saja yang tulis
            y(row) = row_sum;
        }
    });
    
    Kokkos::fence();

    // --- VERIFIKASI ---
    auto h_y = Kokkos::create_mirror_view(y);
    Kokkos::deep_copy(h_y, y);

    printf("Hasil Y: [ ");
    for(int i=0; i<num_rows; i++) printf("%.1f ", h_y(i));
    printf("]\n");
  }
  Kokkos::finalize();
  return 0;
}
