// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define SCAN_BLOCK_SIZE 128

//@@ insert code here
__global__ void cast(float *inputImage, unsigned char *ucharImage, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    ucharImage[i] = (unsigned char)(255 * inputImage[i]);
  }
}

__global__ void rgbtogray(unsigned char *ucharImage, unsigned char *grayImage, long size) {
  long i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    grayImage[i] = 0.21 * ucharImage[3 * i] + 0.71 * ucharImage[3 * i + 1] + 0.07 * ucharImage[3 * i + 2];
  }
}

__global__ void histogram(unsigned char *grayImage, unsigned int *histogram, long size) {
  __shared__ unsigned int private_histogram[256];

  if (threadIdx.x < 256) {
    private_histogram[threadIdx.x] = 0;
  }
  __syncthreads();

  int stride = blockDim.x * gridDim.x;
  long i = blockIdx.x * blockDim.x + threadIdx.x;
  while (i < size) {
    atomicAdd(&(private_histogram[grayImage[i]]), 1);
    i += stride;
  }
  __syncthreads();

  if (threadIdx.x < 256) {
    atomicAdd(&(histogram[threadIdx.x]), private_histogram[threadIdx.x]);
  }
}

__device__ float p(unsigned int x, int len) {
  return float(x) / float(len);
}

// from my MP6
__global__ void scan(unsigned int *input, float *output, int len, long image_size) {
  __shared__ float T[SCAN_BLOCK_SIZE * 2];
  int index = blockDim.x * blockIdx.x*2 + threadIdx.x;
  // load in input numbers
  if (index < len) {
    T[threadIdx.x] = p(input[index], image_size);
  } else {
    T[threadIdx.x] = 0;
  }
  if ((index + blockDim.x) < len) {
    T[threadIdx.x + blockDim.x] = p(input[index + blockDim.x], image_size);
  } else {
    T[threadIdx.x + blockDim.x] = 0;
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
  // // extract block sums
  // if (output != NULL) {
  //   if (threadIdx.x == 0) {
  //     // boundary check for the last block
  //     if (blockDim.x * (blockIdx.x + 1) * 2 > len) {
  //       // does not matter
  //       output[blockIdx.x] = T[len - blockDim.x * blockIdx.x * 2 - 1];
  //     } else {
  //       output[blockIdx.x] = T[blockDim.x * 2 - 1];
  //     }
  //   }
  // }

  // assign back results
  if (index < len) {
    output[index] = T[threadIdx.x];
  } else {
    return;
  }
  if ((index + blockDim.x) < len) {
    output[index + blockDim.x] = T[threadIdx.x + blockDim.x];
  }
}

__device__ float correct_color(float cdfval, float cdfmin) {
  return min(max(255*(cdfval - cdfmin)/(1.0 - cdfmin), 0.0), 255.0);
}

__global__ void equalization(unsigned char *ucharImage, float *outputImage, float *cdf, float cdfmin, long size) {
  long i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < size) {
    outputImage[i] = (float)(correct_color(cdf[ucharImage[i]], cdfmin)/255.0);
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  // print basic information of input image
  printf("width: %d, height: %d, channels: %d\n", imageWidth, imageHeight, imageChannels);
  //@@ insert code here
  float *deviceinputImage;
  unsigned char *deviceucharImage, *devicegrayImage;
  unsigned int *devicehistogram;
  float *devicecdf, *deviceoutputImage;
  // cast from float to unsigned char
  cudaMalloc((void **)&deviceinputImage, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **)&deviceucharImage, imageWidth * imageHeight * imageChannels * sizeof(unsigned char));
  cudaMemcpy((void *)deviceinputImage, (void *)hostInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);
  dim3 dimGrid((imageWidth*imageHeight*imageChannels - 1) / 32 + 1, 1, 1);
  dim3 dimBlock(32, 1, 1);
  cast<<<dimGrid, dimBlock>>>(deviceinputImage, deviceucharImage, imageWidth * imageHeight * imageChannels);
  // print error message
  // cudaError_t err = cudaGetLastError();
  // if (err != cudaSuccess) {
  //   printf("Error: %s\n", cudaGetErrorString(err));
  // }
  cudaFree(deviceinputImage);

  // convert to gray scale
  cudaMalloc((void **)&devicegrayImage, imageWidth * imageHeight * sizeof(unsigned char));
  dimGrid = dim3((imageWidth*imageHeight - 1) / 32 + 1, 1, 1);
  dimBlock = dim3(32, 1, 1);
  rgbtogray<<<dimGrid, dimBlock>>>(deviceucharImage, devicegrayImage, imageWidth * imageHeight);
  // cudaFree(deviceucharImage);

  // compute the histogram
  cudaMalloc((void **)&devicehistogram, HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMemset((void *)devicehistogram, 0, HISTOGRAM_LENGTH * sizeof(unsigned int));
  dimGrid = dim3(16, 1, 1);
  dimBlock = dim3(1024, 1, 1);
  histogram<<<dimGrid, dimBlock>>>(devicegrayImage, devicehistogram, imageWidth * imageHeight);
  cudaFree(devicegrayImage);

  // compute the CDF of histogram
  cudaMalloc((void **)&devicecdf, HISTOGRAM_LENGTH * sizeof(float));
  dimGrid = dim3(1, 1, 1);
  dimBlock = dim3(SCAN_BLOCK_SIZE, 1, 1);
  scan<<<dimGrid, dimBlock>>>(devicehistogram, devicecdf, HISTOGRAM_LENGTH, imageWidth * imageHeight);
  cudaFree(devicehistogram);

  // compute cdfmin
  float *hostcdf = (float *)malloc(HISTOGRAM_LENGTH * sizeof(float));
  cudaMemcpy(hostcdf, devicecdf, sizeof(float) * HISTOGRAM_LENGTH, cudaMemcpyDeviceToHost);
  float cdfmin = hostcdf[0];
  for (int i = 1; i < HISTOGRAM_LENGTH; i++) {
    if (hostcdf[i] < cdfmin) {
      cdfmin = hostcdf[i];
    }
  }
  free(hostcdf);
  // apply equalization
  cudaMalloc((void **)&deviceoutputImage, imageWidth * imageHeight * imageChannels * sizeof(float));
  dimGrid = dim3((imageWidth*imageHeight*imageChannels - 1) / 1024 + 1, 1, 1);
  dimBlock = dim3(1024, 1, 1);
  equalization<<<dimGrid, dimBlock>>>(deviceucharImage, deviceoutputImage, devicecdf, cdfmin, imageWidth*imageHeight*imageChannels);
  cudaFree(deviceucharImage);
  cudaFree(devicecdf);
  cudaMemcpy(hostOutputImageData, deviceoutputImage, sizeof(float) * imageWidth*imageHeight*imageChannels, cudaMemcpyDeviceToHost);
  cudaFree(deviceoutputImage);

  wbImage_setData(outputImage, hostOutputImageData);
  wbSolution(args, outputImage);

  //@@ insert code here

  return 0;
}
