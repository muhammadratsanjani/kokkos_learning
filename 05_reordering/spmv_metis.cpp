#include <Kokkos_Core.hpp>
#include <Kokkos_Timer.hpp>
#include <metis.h> // Wajib install libmetis-dev
#include <vector>
#include <random>
#include <algorithm>
#include <cstdio>
#include <map>

// STRUKTUR DATA UTAMA
struct CSRMatrix {
    std::vector<int> row_map;
    std::vector<int> col_idx;
    std::vector<double> values;
    int num_rows;
    int num_nnz;
};

// 1. GENERATOR MATRIKS (Sama seperti kemarin)
// Kita buat matriks yang agak "blocky" agar efek reordering terasa
// 1. GENERATOR MATRIKS 3D STENCIL (SHUFFLED)
// Ini mensimulasikan masalah fisika nyata (Grid 3D) yang urutan node-nya berantakan.
// METIS harusnya sangat jago membereskan ini.
CSRMatrix generate_3d_stencil_shuffled(int nx, int ny, int nz) {
    int N = nx * ny * nz;
    
    // A. Buat Stencil Standar (7-point) pada Grid Natural
    // Tetangga: (x,y,z) connect to (x±1, y, z), (x, y±1, z), (x, y, z±1)
    std::vector<std::vector<int>> adj(N);
    
    auto get_idx = [&](int x, int y, int z) { return x + y*nx + z*nx*ny; };
    
    for(int z=0; z<nz; z++) {
        for(int y=0; y<ny; y++) {
            for(int x=0; x<nx; x++) {
                int u = get_idx(x,y,z);
                // Tambahkan tetangga (Cek boundary)
                if(x>0) adj[u].push_back(get_idx(x-1, y, z));
                if(x<nx-1) adj[u].push_back(get_idx(x+1, y, z));
                if(y>0) adj[u].push_back(get_idx(x, y-1, z));
                if(y<ny-1) adj[u].push_back(get_idx(x, y+1, z));
                if(z>0) adj[u].push_back(get_idx(x, y, z-1));
                if(z<nz-1) adj[u].push_back(get_idx(x, y, z+1));
            }
        }
    }

    // B. Acak Urutan Node (Simulasi Bad Input)
    // Kita buat peta permutasi acak: old_id -> shuffled_id
    std::vector<int> p(N);
    for(int i=0; i<N; i++) p[i] = i;
    std::mt19937 rng(12345);
    std::shuffle(p.begin(), p.end(), rng);

    // C. Bangun Matriks CSR yang sudah diacak
    CSRMatrix mat;
    mat.num_rows = N;
    mat.row_map.push_back(0);
    int current_nnz = 0;

    // Kita harus memetakan ulang setiap edge u->v menjadi p[u]->p[v]
    // Karena kita butuh baris CSR urut berdasarkan shuffled_id, kita iterasi shuffled_id
    std::vector<int> inv_p(N); // shuffled_id -> old_id
    for(int i=0; i<N; i++) inv_p[p[i]] = i;

    for (int new_u = 0; new_u < N; ++new_u) {
        int old_u = inv_p[new_u]; // Node asli
        
        std::vector<int> new_neighbors;
        for(int old_v : adj[old_u]) {
            new_neighbors.push_back(p[old_v]); // Tetangga di dunia shuffled
        }
        std::sort(new_neighbors.begin(), new_neighbors.end());
        
        for(int col : new_neighbors) {
            mat.col_idx.push_back(col);
            mat.values.push_back(1.0); // Dummy value
            current_nnz++;
        }
        mat.row_map.push_back(current_nnz);
    }
    mat.num_nnz = current_nnz;
    return mat;
}

// 2. FUNGSI BENCHMARK (Running SpMV on GPU/CPU)
double benchmark_spmv(const CSRMatrix& h_mat, int repeat = 100) {
    int N = h_mat.num_rows;
    int NNZ = h_mat.num_nnz;

    // Move to Device
    Kokkos::View<int*>      row_map("row_map", N + 1);
    Kokkos::View<int*>      col_idx("col_idx", NNZ);
    Kokkos::View<double*>   values("values", NNZ);
    Kokkos::View<double*>   x("x", N);
    Kokkos::View<double*>   y("y", N);
    
    // Create Mirrors & Deep Copy (Omitted for brevity, assuming standard way)
    auto h_row_map_v = Kokkos::create_mirror_view(row_map);
    auto h_col_idx_v = Kokkos::create_mirror_view(col_idx);
    auto h_values_v  = Kokkos::create_mirror_view(values);
    auto h_x_v       = Kokkos::create_mirror_view(x);

    for(int i=0; i<=N; i++) h_row_map_v(i) = h_mat.row_map[i];
    for(int i=0; i<NNZ; i++) {
        h_col_idx_v(i) = h_mat.col_idx[i];
        h_values_v(i)  = h_mat.values[i];
    }
    for(int i=0; i<N; i++) h_x_v(i) = 1.0;

    Kokkos::deep_copy(row_map, h_row_map_v);
    Kokkos::deep_copy(col_idx, h_col_idx_v);
    Kokkos::deep_copy(values, h_values_v);
    Kokkos::deep_copy(x, h_x_v);

    Kokkos::fence();
    Kokkos::Timer timer;
    
    for(int iter=0; iter<repeat; iter++) {
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
    Kokkos::fence();
    return timer.seconds() / repeat;
}

// 3. FUNGSI PERMUTASI (Permute Matrix Rows & Cols based on Map)
// Ini bagian tersulit: Mengacak ulang matriks berdasarkan "peta" baru.
CSRMatrix permute_matrix(const CSRMatrix& src, const std::vector<idx_t>& perm) {
    int N = src.num_rows;
    CSRMatrix dest;
    dest.num_rows = N;
    dest.num_nnz = src.num_nnz;
    dest.row_map.push_back(0);

    // perm[old_row] = new_row_id
    // Tapi kita butuh kebalikannya: iperm[new_row_id] = old_row
    std::vector<int> iperm(N);
    for(int i=0; i<N; i++) iperm[perm[i]] = i;

    int current_nnz = 0;
    for (int new_row = 0; new_row < N; ++new_row) {
        int old_row = iperm[new_row]; // Baris mana yang harus ditaruh di sini?
        
        int start = src.row_map[old_row];
        int end   = src.row_map[old_row+1];
        
        // Kita juga harus merename col_idx sesuai permutasi agar simetris
        // (Asumsi matriks kotak & permutasi simetris P * A * P^T)
        // Kalau cuma reorder baris, cache x vector tetap berantakan.
        std::vector<int> new_cols;
        std::vector<double> new_vals;

        for(int k=start; k<end; k++) {
             int old_col = src.col_idx[k];
             int new_col = perm[old_col]; // Rename column ID juga!
             new_cols.push_back(new_col);
             new_vals.push_back(src.values[k]);
        }

        // Jangan lupa sort kolom lagi (wajib untuk CSR)
        // Kita pakai pair sort coding manual atau struct kecil
        std::vector<std::pair<int, double>> temp;
        for(size_t j=0; j<new_cols.size(); j++) temp.push_back({new_cols[j], new_vals[j]});
        std::sort(temp.begin(), temp.end());

        for(auto& p : temp) {
            dest.col_idx.push_back(p.first);
            dest.values.push_back(p.second);
            current_nnz++;
        }
        dest.row_map.push_back(current_nnz);
    }
    return dest;
}


int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    // Gunakan Grid 3D: 80x80x80 = 512.000 Node
    const int GRID_DIM = 150; 
    const int N = GRID_DIM * GRID_DIM * GRID_DIM; 
    printf("Experiment: METIS Ordering Effect on SpMV (3D Stencil)\n");
    printf("Matrix Size: %d x %d (from %d^3 Grid)\n", N, N, GRID_DIM);

    // A. Generate "Bad" Matrix (Shuffled Grid)
    printf("Generating Shuffled 3D Grid...\n");
    CSRMatrix mat_orig = generate_3d_stencil_shuffled(GRID_DIM, GRID_DIM, GRID_DIM);
    double t_orig = benchmark_spmv(mat_orig);
    printf("[Baseline] Original Time: %f s | %.2f GFLOPs\n", 
           t_orig, (2.0*mat_orig.num_nnz*1e-9)/t_orig);

    // B. Calculate Reordering using METIS
    // METIS butuh adjancency structure. Untuk matriks simetris, CSR row_map/col_idx mirip adjancency.
    // Untuk non-simetris random, ini agak hacky, tapi kita coba anggap graph tak berarah.
    
    idx_t n_metis = N;
    idx_t ncon = 1;
    
    // Siapkan array METIS (harus tipe idx_t)
    std::vector<idx_t> xadj(mat_orig.row_map.begin(), mat_orig.row_map.end());
    std::vector<idx_t> adjncy(mat_orig.col_idx.begin(), mat_orig.col_idx.end());
    std::vector<idx_t> perm(N);  // Output: New ID for each node
    std::vector<idx_t> iperm(N); // Output: Old ID for each new position

    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);

    printf("Running METIS NodeND (Nested Dissection)...\n");
    int status = METIS_NodeND(&n_metis, xadj.data(), adjncy.data(), NULL, options, perm.data(), iperm.data());

    if(status != METIS_OK) {
        printf("METIS Error! Code: %d\n", status);
    } else {
        // C. Permute Matrix
        CSRMatrix mat_opt = permute_matrix(mat_orig, perm);
        
        // D. Benchmark Optimized Matrix
        double t_opt = benchmark_spmv(mat_opt);
        printf("[Optimized] METIS Time : %f s | %.2f GFLOPs\n", 
               t_opt, (2.0*mat_opt.num_nnz*1e-9)/t_opt);
               
        double speedup = t_orig / t_opt;
        printf(">>> SPEEDUP: %.2fx <<<\n", speedup);
    }

  }
  Kokkos::finalize();
  return 0;
}
