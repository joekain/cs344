/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"

__global__
void min_step(const float* const input,
              float* output,
              int input_size)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (2 * i >= input_size) return;

  float a = input[2 * i];

  if (2 * i + 1 < input_size) {
    float b = input[2 * i + 1];
    output[i] = min(a, b);
  } else {
    output[i] = a;
    }
}

__global__
void max_step(const float* const input,
              float* output,
              int input_size)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (2 * i >= input_size) return;

  float a = input[2 * i];

  if (2 * i + 1 < input_size) {
    float b = input[2 * i + 1];
    output[i] = max(a, b);
  } else {
    output[i] = a;
  }
}

__global__
void histogram(const float *const input,
               unsigned int *bins, int numBins,
               float min, float range)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int bin = (input[i] - min) / range * numBins;
  atomicAdd(&bins[bin], 1);
}

void reduce_min_max(const float *const d_logLuminance,
                    float &min_logLum,
                    float &max_logLum,
                    const size_t numRows,
                    const size_t numCols)
{
  float *t1, *t2;

  size_t initial_size = numRows * numCols;

  checkCudaErrors(cudaMalloc(&t1, sizeof(float) * initial_size));
  checkCudaErrors(cudaMalloc(&t2, sizeof(float) * initial_size));

  const float * src = d_logLuminance;
  float *dst = t2;
  size_t input_size;
  size_t output_size;
  for (input_size = initial_size; input_size >= 2; input_size = output_size) {
    output_size = (input_size + 1) / 2;
    min_step<<<dim3((output_size + 63) / 64), dim3(64)>>>(src, dst, input_size);
    src = dst;
    if (src == t1) {
      dst = t2;
    } else {
      dst = t1;
    }
  }
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // Result is in src because we exchanged src and tmp after the last kernel launch
  checkCudaErrors(cudaMemcpy(&min_logLum, src, sizeof(float), cudaMemcpyDeviceToHost));
  printf("min_logLum = %f\n", min_logLum);

  src = d_logLuminance;
  dst = t2;
  for (input_size = initial_size; input_size >= 2; input_size = output_size) {
    output_size = (input_size + 1) / 2;
    max_step<<<dim3((output_size + 63) / 64), dim3(64)>>>(src, dst, input_size);
    src = dst;
    if (src == t1) {
      dst = t2;
    } else {
      dst = t1;
    }
  }
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  // Result is in src because we exchanged src and tmp after the last kernel launch
  checkCudaErrors(cudaMemcpy(&max_logLum, src, sizeof(float), cudaMemcpyDeviceToHost));
  printf("max_logLum = %f\n", max_logLum);

  cudaFree(t1);
  cudaFree(t2);
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
    store in min_logLum and max_logLum */
  reduce_min_max(d_logLuminance, min_logLum, max_logLum, numRows, numCols);

  /*2) subtract them to find the range */
  float logLumRange = max_logLum - min_logLum;


  /*3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins */
  unsigned int *d_bins;
  checkCudaErrors(cudaMalloc(&d_bins, sizeof(unsigned int) * numBins));
  histogram<<<dim3(numRows * numCols), dim3(1)>>>(d_logLuminance,
                                                  d_bins, numBins,
                                                  min_logLum, logLumRange);
  
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  unsigned int histo0;
  checkCudaErrors(cudaMemcpy(&histo0, d_bins, sizeof(unsigned), cudaMemcpyDeviceToHost));
  printf("histo[0] = %d\n", histo0);

  //TODO
  /*4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
  cudaFree(d_bins);
}
