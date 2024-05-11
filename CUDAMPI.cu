#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <endian.h>
#include <string.h>
#include <CUDAMPI.h>

__global__ void conv_forward_kernel(double* input, double* output, double* weights, double* biases,
                                    int inputWidth, int inputHeight, int inputDepth,
                                    int outputWidth, int outputHeight, int outputDepth,
                                    int kernelSize, int padding, int stride) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;
    int tz = threadIdx.z + blockIdx.z * blockDim.z;

    if (tx < outputWidth && ty < outputHeight && tz < outputDepth) {
        double value = 0.0;
        int from_x = tx * stride - padding;
        int from_y = ty * stride - padding;
        for (int i = 0; i < kernelSize; ++i) {
            for (int j = 0; j < kernelSize; ++j) {
                for (int k = 0; k < inputDepth; ++k) {
                    int ix = from_x + i;
                    int iy = from_y + j;
                    if (ix >= 0 && ix < inputWidth && iy >= 0 && iy < inputHeight) {
                        int input_idx = (k * inputHeight + iy) * inputWidth + ix;
                        int weight_idx = ((tz * inputDepth + k) * kernelSize + j) * kernelSize + i;
                        value += input[input_idx] * weights[weight_idx];
                    }
                }
            }
        }
        int bias_idx = tz;
        output[(tz * outputHeight + ty) * outputWidth + tx] = value + biases[bias_idx];
    }
}


void forward_convolution_layer(Layer* layer) {
    double* d_output;
    cudaMalloc(&d_output, sizeof(double) * layer->depth * layer->width * layer->height);
    double* d_weights;
    double* d_biases;
    cudaMalloc(&d_weights, sizeof(double) * layer->nweights);
    cudaMalloc(&d_biases, sizeof(double) * layer->nbiases);
    cudaMemcpy(d_weights, layer->weights, sizeof(double) * layer->nweights, cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, layer->biases, sizeof(double) * layer->nbiases, cudaMemcpyHostToDevice);
    dim3 blockDim(16, 16, 1);
    dim3 gridDim((layer->width  + 15) / 16, (layer->height + 15) / 16, layer->depth);
    conv_forward_kernel<<<gridDim, blockDim>>>(layer->lprev->output, d_output, d_weights, d_biases,
                                               layer->prev->width, layer->prev->height, layer->prev->depth,
                                               layer->width, layer->height, layer->depth,
                                               layer->conv.kernsize, layer->conv.padding, layer->conv.stride);

    cudaMemcpy(layer->outputs, d_output, sizeof(double) * layer->depth * layer->width * layer->height, cudaMemcpyDeviceToHost);
    cudaFree(d_output);
    cudaFree(d_weights);
    cudaFree(d_biases);
}