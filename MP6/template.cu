// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 1024 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void add_blocksum(float *scanned_blocks, float *scanned_blocksum, float *output, int len) {
  int index = blockIdx.x * blockDim.x*2 + threadIdx.x;
  if (index < len) {
    if (blockIdx.x >= 1) {
      int blocksum = scanned_blocksum[blockIdx.x - 1];
      output[index] = blocksum + scanned_blocks[index];
    } else {
      output[index] = scanned_blocks[index];
    }
  }
  index += blockDim.x;
  if (index < len) {
    if (blockIdx.x >= 1) {
      int blocksum = scanned_blocksum[blockIdx.x - 1];
      output[index] = blocksum + scanned_blocks[index];
    } else {
      output[index] = scanned_blocks[index];
    }
  }
}

__global__ void scan(float *input, float *output, int len) {
// __global__ void scan(float *input, float *output, float *blocksum, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float T[BLOCK_SIZE * 2];
  int index = blockDim.x * blockIdx.x*2 + threadIdx.x;
  // load in input numbers
  if (index < len) {
    T[threadIdx.x] = input[index];
  } else {
    T[threadIdx.x] = 0.0f;
  }
  if ((index + blockDim.x) < len) {
    T[threadIdx.x + blockDim.x] = input[index + blockDim.x];
  } else {
    T[threadIdx.x + blockDim.x] = 0.0f;
  }
  // Prescan
  int stride = 1;
  while (stride < 2*blockDim.x) {
    __syncthreads();
    index = (threadIdx.x + 1)*stride*2 - 1;
    if (index < 2*blockDim.x && index >= stride) {
      T[index] += T[index - stride];
    }
    stride *= 2;
  }

  // Postscan
  stride = blockDim.x / 2;
  while (stride > 0) {
    __syncthreads();
    index = (threadIdx.x + 1)*stride*2 - 1;
    if ((index + stride) < 2*blockDim.x) {
      T[index + stride] += T[index];
    }
    stride /= 2;
  }

  __syncthreads();
  index = blockDim.x * blockIdx.x*2 + threadIdx.x;
  // extract block sums
  if (output != NULL) {
    if (threadIdx.x == 0) {
      // boundary check for the last block
      if (blockDim.x * (blockIdx.x + 1) * 2 > len) {
        // does not matter
        output[blockIdx.x] = T[len - blockDim.x * blockIdx.x * 2 - 1];
      } else {
        output[blockIdx.x] = T[blockDim.x * 2 - 1];
      }
    }
  }

  // assign back results
  if (index < len) {
    input[index] = T[threadIdx.x];
  } else {
    return;
  }
  if ((index + blockDim.x) < len) {
    input[index + blockDim.x] = T[threadIdx.x + blockDim.x];
  }
}

void savefile(float *input, int len, char *name) {
  // save input from GPU to file
  FILE *f = fopen(name, "w");
  float temp[1000];
  cudaMemcpy(temp, input, len * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < len; i++) {
    fprintf(f, "%f\n", temp[i]);
  }
  fclose(f);
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid((numElements - 1)/(BLOCK_SIZE*2) + 1, 1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  float *blocksum;
  cudaMalloc((void **)&blocksum, dimGrid.x * sizeof(float));
  // cudaMemset(blocksum, 0.0f, dimGrid.x);
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  // scan inside blocks of data
  // scan<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, blocksum, numElements);
  scan<<<dimGrid, dimBlock>>>(deviceInput, blocksum, numElements);

  // perform scan on the array
  cudaDeviceSynchronize();
  // savefile(deviceInput, numElements, "scan1.txt");
  // savefile(blocksum, dimGrid.x, "blocksum.txt");
  // scan<<<1, dimBlock>>>(blocksum, deviceOutput, NULL, dimGrid.x);
  scan<<<1, dimBlock>>>(blocksum, NULL, dimGrid.x);
  // add the sum of previous blocks (no need to perform on the first block)
  add_blocksum<<<dimGrid, dimBlock>>>(deviceInput, blocksum, deviceOutput, numElements); 

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(blocksum);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
