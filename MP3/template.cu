
#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  const int TILE_WIDTH = 32;
  __shared__ float subTileA[32][TILE_WIDTH];
  __shared__ float subTileB[TILE_WIDTH][32];

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int colomn = blockIdx.x * blockDim.x + threadIdx.x;
  float Cvalue = 0;

  for (int m = 0; m < (numAColumns - 1)/TILE_WIDTH + 1; ++m) {
    if ((m * TILE_WIDTH + threadIdx.x) >= numAColumns || row >= numARows) {
      subTileA[threadIdx.y][threadIdx.x] = 0;
    } else {
      subTileA[threadIdx.y][threadIdx.x] = A[row * numAColumns + m * blockDim.x + threadIdx.x];
    }
    if ((m * TILE_WIDTH + threadIdx.y) >= numBRows || colomn >= numBColumns) {
      subTileB[threadIdx.y][threadIdx.x] = 0;
    } else {
      subTileB[threadIdx.y][threadIdx.x] = B[colomn + (m * blockDim.y + threadIdx.y) * numBColumns];
    }
    __syncthreads();
    if (row < numCRows && colomn < numCColumns) {
      for (int k = 0; k < TILE_WIDTH; ++k) {
        Cvalue += subTileA[threadIdx.y][k] * subTileB[k][threadIdx.x];
      }
    }
    __syncthreads();
  }
  if (row < numCRows && colomn < numCColumns) {
    C[row * numCColumns + colomn] = Cvalue;
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)
  int sizeA, sizeB, sizeC;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
  sizeA = sizeof(float) * numAColumns * numARows;
  sizeB = sizeof(float) * numBColumns * numBRows;
  sizeC = sizeof(float) * numCColumns * numCRows;
  //@@ Allocate the hostC matrix
  hostC = (float *)malloc(sizeC);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);



  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc(&deviceA, sizeA);
  cudaMalloc(&deviceB, sizeB);
  cudaMalloc(&deviceC, sizeC);

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, sizeB, cudaMemcpyHostToDevice);

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid(ceil(numCColumns/32.0), ceil(numCRows/32.0), 1);
  dim3 DimBlock(32, 32, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<DimGrid, DimBlock>>>(deviceA, deviceB, deviceC, numARows,
                                        numAColumns, numBRows, numBColumns,
                                        numCRows, numCColumns);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, sizeC, cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
