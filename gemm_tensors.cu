#include <stdio.h>
#include <mma.h>

using namespace nvcuda;

// Must be multiples of 16
#define MATRIX_M 16384
#define MATRIX_N 16384
#define MATRIX_K 16384

// Dimension of WMMA warp level matrices
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

__global__ void gemm_tensor(half *a, half *b, float *c, int M, int N, int K,
                            float alpha, float beta) {
  // Leading dimensions
  int lda = M;
  int ldb = K;
  int ldc = M;

  // Tile using a 2D grid
  // warpSize is 32, so converting Global Thread ID into Global Warp ID  for M-dimension
  int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  // we don't divide the N-dimension by warpSize due to how blockDim.x and blockDim.y are set (128,4)
  int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

  // Declare framents
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

  // Set acc_frag to 0.0
  wmma::fill_fragment(acc_frag, 0.0f);

  // The strategy for the GEMM is to compute one tile of 16 x 16 output matrix per warp.
  for (int i = 0; i < K; i += WMMA_K) {
    int aRow = warpM * WMMA_M;
    int aCol = i;
    int bRow = i;
    int bCol = warpN * WMMA_N;

    // Bounds checking
    if ( aRow < M && aCol < K && bRow < K && bCol < N) {
      // Load the inputs
      // The load_matrix_sync function is hardcoded to go the memory address of pointer
      // and grad a whole 16 x 16 block, and distribute among the registers of
      // 32 threads in the warp
      wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda , lda);
      wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);

      // Perform the matrix multiplication and returns the 16 x 16 output tile
      wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
      // _sync suggests to wait until all 32 threads in a warp have finished

    }
  }
    // Scale with alpha and beta
    // Load in current value of c, scale by beta, and add to result scaled by alpha
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;

    if (cRow < M && cCol < N) {
      wmma::load_matrix_sync(c_frag, c + cRow + cCol * ldc, ldc, wmma::mem_row_major);

      for (int i = 0; i < c_frag.num_elements; i++) {
        // Element wise addition
        c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
      }

      // Store the output
    wmma::store_matrix_sync(c + cRow + cCol * ldc, c_frag, ldc, wmma::mem_row_major);
    }
}

int main() {
  return 0;
}
