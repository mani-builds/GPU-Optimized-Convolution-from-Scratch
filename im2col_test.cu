#include <stdio.h>
#include <math.h>

#define STRIDE 1
#define FILTER_RADIUS 1
#define PADDING FILTER_RADIUS

// __device__ float im2col[im2col_H][im2col_W];

#define IDX2R(i,j,width) ((i)*(width) + (j))
#define IDX2C(i,j,width) ((j)*(width) + (i))

__global__ void im2col_kernel(float *image,int C_in, int H_in, int W_in, int im2col_H, int im2col_W, int kernel_size, float *im2col){
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // If the input is 5x5 and kernel is 3x3 then im2col will be 9x25
  // 9 is the patch size and 25 is number of patches
  // Each thread could create a patch, so in total there are 25 patches = 5x5

  // int patch_idx = out_row * im2col_W + out_col;
  int idx_count = 0;

  if (row < H_in && col < W_in) {
    for (int c=0; c<C_in; c++){
    for (int i = -FILTER_RADIUS; i <= FILTER_RADIUS; i++) {
      for (int j = -FILTER_RADIUS; j <= FILTER_RADIUS; j++) {
        float val = 0.0f;
        if (col + j >= 0 && col + j < W_in && row + i >= 0 && row + i < H_in) {
          val  = image[c*(H_in*W_in) + (row+i) * W_in + (col+j)];
        }
        im2col[IDX2R(idx_count, row * W_in + col, im2col_W)] = val;
        idx_count++;
      }
    }
    }
  }
}

int output_dim(int size_of_input, int padding, int kernel_size, int stride){
  return floor((size_of_input + 2 * padding - kernel_size) / stride) + 1;
}

int main(){
  int kernel_size = 2*FILTER_RADIUS + 1;
  int H_in, W_in;
  int C_in = 3;

  int H_out, W_out;
  int im2col_H, im2col_W;
  int C_out = C_in;

  H_in = 5;
  W_in = 5;

  float *image_h;
  float *kernel_h;
  float *im2col_h;

  image_h = (float *)malloc(C_in*H_in*W_in*sizeof(float));
  kernel_h = (float *)malloc(C_in*kernel_size*kernel_size*sizeof(float));


  for (int i = 0; i < C_in * H_in * W_in; i++) {
    image_h[i] = i;
  }
  for (int i = 0; i < C_in * kernel_size * kernel_size; i++) {
    kernel_h[i] = i;
  }

  printf("\nInput image: \n");
  for (int i = 0; i < C_in * H_in * W_in; i++) {
    printf("%.0f ", image_h[i]);
  }
  printf("\nKernel: \n");
  for (int i = 0; i < C_in * kernel_size * kernel_size; i++) {
    printf("%.0f ", kernel_h[i]);
  }

  H_out = output_dim(H_in, PADDING, kernel_size, STRIDE);
  W_out = output_dim(W_in, PADDING, kernel_size, STRIDE);

  im2col_H = C_in * kernel_size * kernel_size;
  im2col_W = H_out * W_out;
  im2col_h = (float *)malloc(im2col_H*im2col_W*sizeof(float));

  float *image;
  float *kernel;
  float *im2col;

  cudaMalloc(&image, C_in*H_in*W_in*sizeof(float));
  cudaMalloc(&kernel, C_in*kernel_size*kernel_size*sizeof(float));
  cudaMalloc(&im2col, im2col_H*im2col_W*sizeof(float));

  cudaMemcpy(image,image_h,C_in*H_in*W_in*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(kernel,kernel_h,C_in*kernel_size*kernel_size*sizeof(float), cudaMemcpyHostToDevice);

  printf("\nInput DIM H_in and W_in: (%d, %d)\n",H_in, W_in);
  printf("\nim2col DIM im2col_H and im2col_W: (%d, %d)\n",im2col_H, im2col_W);

  dim3 threadsPerBlock(H_out,  W_out);
  dim3 blocksPerGrid((C_out * H_out + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (W_out + threadsPerBlock.y - 1) / threadsPerBlock.y);

  im2col_kernel<<<blocksPerGrid,threadsPerBlock>>>(image,C_in, H_in, W_in, im2col_H, im2col_W, kernel_size, im2col);

  cudaMemcpy(im2col_h, im2col, im2col_H*im2col_W*sizeof(float), cudaMemcpyDeviceToHost);

  printf("\nIm2col image: \n");
  for (int i = 0; i < im2col_H ; i++) {
  for (int j = 0; j < im2col_W ; j++) {
    printf("%.0f ", im2col_h[IDX2R(i, j, im2col_W)]);
  }
  printf("\n");
  }

  free(image_h);
  free(kernel_h);
  free(im2col_h);
  cudaFree(image);
  cudaFree(kernel);
  cudaFree(im2col);

}
