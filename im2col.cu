#include <stdio.h>
#include <math.h>

#define STRIDE 1
#define FILTER_RADIUS 1
#define PADDING FILTER_RADIUS

// __device__ float im2col[im2col_H][im2col_W];

#define IDX2R(i,j,width) ((i)*(width) + (j))
#define IDX2C(i,j,width) ((j)*(width) + (i))

__global__ void im2col_kernel(float *image, int C_in, int H_in, int W_in,
                              int H_out, int W_out, int im2col_H, int im2col_W, int kernel_size, float *im2col){
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
        im2col[IDX2R(idx_count, row * W_out + col, im2col_W)] = val;
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
  int C_in;

  int H_out, W_out;
  int im2col_H, im2col_W;
  int C_out;

  char f_input_array_shape[50];
  printf("\n Enter the file of input image array shape (txt): ");
  scanf("%s", f_input_array_shape);
  FILE *file = fopen(f_input_array_shape, "r");
    if (file == NULL) {
        return 1; // Error opening file
    }
    float value;
    float *image_shape;
    image_shape = (float *)malloc(sizeof(float)*4);
    int i =0;
    while (fscanf(file, "%f", &value) == 1) {
      image_shape[i] = value;
        i++;
    }

    // printf("Image shape sample: \n");
    // for(int i=0; i<3; i++) printf("%f\t ", image_shape[i]);
    // printf("\n");


  char f_kernel_array_shape[50];
  printf("\n Enter the file of kernel array shape (txt): ");
  scanf("%s", f_kernel_array_shape);
    FILE *kfile = fopen(f_kernel_array_shape, "r");
    if (kfile == NULL) {
        return 1; // Error opening file
    }
    float svalue;
    float *kernel_shape;
    kernel_shape = (float *)malloc(sizeof(float)*3);
    i =0;
    while (fscanf(kfile, "%f", &svalue) == 1) {
        kernel_shape[i] = svalue;
        i++;
    }

    // printf("kernel shape sample: \n");
    // for(int i=0; i<3; i++) printf("%f\t ", kernel_shape[i]);
    // printf("\n");

  C_in = image_shape[0];
  H_in = image_shape[1];
  W_in = image_shape[2];

  C_out = kernel_shape[0];
  kernel_size = kernel_shape[2];

  float *image_h;
  // float *kernel_h;
  float *im2col_h;

  image_h = (float *)malloc(C_in*H_in*W_in*sizeof(float));
  // kernel_h = (float *)malloc(C_out*C_in*kernel_size*kernel_size*sizeof(float));

  char f_input_image_array[50];
  printf("\n Enter the file of image array (txt): ");
  scanf("%s", f_input_image_array);
  FILE *image_file = fopen(f_input_image_array, "r");
  if (image_file == NULL) {
        return 1; // Error opening file
    }
    i=0;
    float ivalue;
    while (fscanf(image_file, "%f", &ivalue) == 1) {
      image_h[i] = ivalue;
      i++;
    }

    // printf("Image array sample: \n");
    // for(int i=0; i<10; i++) printf("%f\t ", image_h[i]);
    // printf("\n");

  // char f_kernel_array[50];
  // printf("\n Enter the file name of kernel array (txt): ");
  // scanf("%s", f_kernel_array);
  // FILE *kernel_file = fopen(f_kernel_array, "r");
  // if (kernel_file == NULL) {
  //       return 1; // Error opening file
  //   }
  //   i=0;
  //   float kvalue;
  //   while (fscanf(kernel_file, "%f", &kvalue) == 1) {
  //     kernel_h[i] = kvalue;
  //     i++;
  //   }

    // printf("Kernel array sample: \n");
    // for(int i=0; i<10; i++) printf("%f\t ", kernel_h[i]);
    // printf("\n");

  H_out = output_dim(H_in, PADDING, kernel_size, STRIDE);
  W_out = output_dim(W_in, PADDING, kernel_size, STRIDE);

  im2col_H = C_in * kernel_size * kernel_size;
  im2col_W = H_out * W_out;
  im2col_h = (float *)malloc(im2col_H*im2col_W*sizeof(float));

  float *image;
  // float *kernel;
  float *im2col;

  cudaMalloc(&image, C_in*H_in*W_in*sizeof(float));
  // cudaMalloc(&kernel, C_out*C_in*kernel_size*kernel_size*sizeof(float));
  cudaMalloc(&im2col, im2col_H*im2col_W*sizeof(float));

  cudaMemcpy(image,image_h,C_in*H_in*W_in*sizeof(float), cudaMemcpyHostToDevice);
  // cudaMemcpy(kernel,kernel_h,C_out*C_in*kernel_size*kernel_size*sizeof(float), cudaMemcpyHostToDevice);

  printf("\nInput DIM H_in and W_in: (%d, %d)",H_in, W_in);
  printf("\nOutput DIM H_out and W_out: (%d, %d)",H_out, W_out);
  printf("\nim2col DIM im2col_H and im2col_W: (%d, %d)\n",im2col_H, im2col_W);

  dim3 threadsPerBlock(16,  16);
  dim3 blocksPerGrid((W_in + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (H_in + threadsPerBlock.y - 1) / threadsPerBlock.y);

  im2col_kernel<<<blocksPerGrid, threadsPerBlock>>>(
      image, C_in, H_in, W_in, H_out, W_out, im2col_H, im2col_W,
                                                       kernel_size, im2col);
    // // Check for immediate launch errors
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     printf("CUDA Error: %s\n", cudaGetErrorString(err));
    // }

    // // Wait for kernel to finish and check for execution errors
    // err = cudaDeviceSynchronize();
    // if (err != cudaSuccess) {
    //     printf("CUDA Kernel Execution Error: %s\n", cudaGetErrorString(err));
    // }
  cudaMemcpy(im2col_h, im2col, im2col_H*im2col_W*sizeof(float), cudaMemcpyDeviceToHost);

  char f_output_im2col_array[50];
  printf("\n Enter the file name of im2col output array (txt): ");
  scanf("%s", f_output_im2col_array);
  FILE *im2co_file = fopen(f_output_im2col_array, "w");
  printf("\nWriting the Im2col image \n");
  for (int i = 0; i < im2col_H ; i++) {
  for (int j = 0; j < im2col_W ; j++) {
    fprintf(im2co_file, "%.0f ", im2col_h[IDX2R(i, j, im2col_W)]);
  }
  }

  fclose(file);
  fclose(image_file);
  // fclose(kernel_file);
  fclose(im2co_file);
  free(image_shape);
  free(image_h);
  // free(kernel_h);
  free(im2col_h);
  cudaFree(image);
  // cudaFree(kernel);
  cudaFree(im2col);

}
