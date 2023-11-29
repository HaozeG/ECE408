#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
const int MASK_WIDTH = 3;
const int TILE_WIDTH = 6;

//@@ Define constant memory for device kernel here
__constant__ float Mc[MASK_WIDTH][MASK_WIDTH][MASK_WIDTH];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  // Declare shared memory
  __shared__ float tile[TILE_WIDTH + MASK_WIDTH - 1][TILE_WIDTH + MASK_WIDTH - 1][TILE_WIDTH + MASK_WIDTH - 1];
  // Get data to shared memory
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;

  int y_o = blockIdx.y * TILE_WIDTH + ty;
  int x_o = blockIdx.x * TILE_WIDTH + tx;
  int z_o = blockIdx.z * TILE_WIDTH + tz;

  int y_i = y_o - 1;  // -1: MASK_WIDTH/2
  int x_i = x_o - 1;
  int z_i = z_o - 1;

  if ((x_i >= 0) && (x_i < x_size) && (y_i >= 0) && (y_i < y_size) && (z_i >= 0) && (z_i < z_size)) {
    tile[tz][ty][tx] = input[z_i * x_size * y_size + y_i * x_size + x_i];
  } else {
    tile[tz][ty][tx] = 0.0f;
  }
  __syncthreads();
  // Compute
  if ((tx < TILE_WIDTH) && (ty < TILE_WIDTH) && (tz < TILE_WIDTH)) {
    if ((x_o < x_size) && (y_o < y_size) && (z_o < z_size)) {
      float Pvalue = 0.0f;
      for (int i = 0; i < MASK_WIDTH; ++i) {
        for (int j = 0; j < MASK_WIDTH; ++j) {
          for (int k = 0; k < MASK_WIDTH; ++k) {
            Pvalue += Mc[i][j][k] * tile[tz + i][ty + j][tx + k];
          }
        }
      }
      output[z_o * x_size * y_size + y_o * x_size + x_o] = Pvalue;
    }
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  cudaMalloc(&deviceInput, sizeof(float) * (inputLength - 3));
  cudaMalloc(&deviceOutput, sizeof(float) * (inputLength - 3));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(deviceInput, hostInput + 3, sizeof(float) * (inputLength - 3), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(Mc, hostKernel, kernelLength * sizeof(float));
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 DimGrid(ceil(x_size/(1.0*TILE_WIDTH)), ceil(y_size/(1.0*TILE_WIDTH)), ceil(z_size/(1.0*TILE_WIDTH)));
  dim3 DimBlock(TILE_WIDTH + MASK_WIDTH - 1, TILE_WIDTH + MASK_WIDTH - 1, TILE_WIDTH + MASK_WIDTH - 1);

  //@@ Launch the GPU kernel here
  conv3d<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(hostOutput + 3, deviceOutput, sizeof(float) * (inputLength - 3), cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");
  // output hostOutput into a file
  // float* hostOutput;
  // int outputLength = inputLength;

  // Allocate memory on the host
  // hostOutput = (float*)malloc(sizeof(float) * outputLength);

  // ... Code to fill the hostOutput array ...

  


  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Open the output file
  FILE* outputFile = fopen("/home/haoze/ECE408/MP4/output.txt", "w");

  // Write the output data to the file
  for (int i = 0; i < inputLength; i++) {
      fprintf(outputFile, "%f/n", hostOutput[i]);
  }

  // Close the output file
  fclose(outputFile);
  

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  // delay
  sleep(10);
  return 0;
}
