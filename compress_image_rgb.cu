#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

// Optimized kernel to compute row sums and sum of squares using shared memory
__global__ void compute_row_stats(float *X, float *S_row, float *S_row_sq, int M, int N) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    extern __shared__ float sdata[];
    float *s_sum = sdata;
    float *s_sum_sq = &sdata[blockDim.x];
    
    if (row < M) {
        float sum = 0.0f;
        float sum_sq = 0.0f;
        
        // Each thread processes multiple elements
        for (int col = tid; col < N; col += blockDim.x) {
            float val = X[row * N + col];
            sum += val;
            sum_sq += val * val;
        }
        
        s_sum[tid] = sum;
        s_sum_sq[tid] = sum_sq;
        __syncthreads();
        
        // Reduction
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                s_sum[tid] += s_sum[tid + stride];
                s_sum_sq[tid] += s_sum_sq[tid + stride];
            }
            __syncthreads();
        }
        
        if (tid == 0) {
            S_row[row] = s_sum[0];
            S_row_sq[row] = s_sum_sq[0];
        }
    }
}

// Optimized kernel to compute column sums and sum of squares using shared memory
__global__ void compute_col_stats(float *X, float *S_col, float *S_col_sq, int M, int N) {
    int col = blockIdx.x;
    int tid = threadIdx.x;
    
    extern __shared__ float sdata[];
    float *s_sum = sdata;
    float *s_sum_sq = &sdata[blockDim.x];
    
    if (col < N) {
        float sum = 0.0f;
        float sum_sq = 0.0f;
        
        // Each thread processes multiple elements
        for (int row = tid; row < M; row += blockDim.x) {
            float val = X[row * N + col];
            sum += val;
            sum_sq += val * val;
        }
        
        s_sum[tid] = sum;
        s_sum_sq[tid] = sum_sq;
        __syncthreads();
        
        // Reduction
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                s_sum[tid] += s_sum[tid + stride];
                s_sum_sq[tid] += s_sum_sq[tid + stride];
            }
            __syncthreads();
        }
        
        if (tid == 0) {
            S_col[col] = s_sum[0];
            S_col_sq[col] = s_sum_sq[0];
        }
    }
}

// Helper function to print matrix in readable format (integers only)
void print_matrix(const char* title, float* matrix, int rows, int cols) {
    std::cout << "\n" << title << " (" << rows << "x" << cols << "):\n";
    std::cout << std::string(cols * 5 + 2, '=') << "\n";
    
    for (int i = 0; i < rows; i++) {
        std::cout << " ";
        for (int j = 0; j < cols; j++) {
            printf("%4.0f ", matrix[i * cols + j]); // Show as integers
        }
        std::cout << "\n";
    }
    std::cout << std::string(cols * 5 + 2, '=') << "\n";
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <input_image> <output_compressed_file>\n";
        std::cout << "  input_image: PNG, JPG, BMP, or TGA image file\n";
        std::cout << "  output_compressed_file: Output file for compressed representation\n";
        std::cout << "\nSupported formats: PNG, JPG, JPEG, BMP, TGA\n";
        std::cout << "Supports both grayscale and RGB images\n";
        return 1;
    }

    const char* input_file = argv[1];
    const char* output_file = argv[2];

    // Load image using stb_image
    int width, height, channels;
    unsigned char* image_data = stbi_load(input_file, &width, &height, &channels, 0); // Keep original channels
    
    if (!image_data) {
        std::cerr << "Error: Cannot load image " << input_file << std::endl;
        std::cerr << "Supported formats: PNG, JPG, JPEG, BMP, TGA" << std::endl;
        return 1;
    }

    int M = height; // rows
    int N = width;  // columns
    int C = channels; // channels (1 for grayscale, 3 for RGB, 4 for RGBA)
    
    // For RGBA, treat as RGB (ignore alpha channel)
    if (C == 4) C = 3;
    
    // For unsupported channel counts, convert to grayscale
    if (C != 1 && C != 3) {
        std::cout << "Warning: " << channels << " channels not fully supported, converting to grayscale" << std::endl;
        C = 1;
        unsigned char* gray_data = new unsigned char[M * N];
        for (int i = 0; i < M * N; i++) {
            // Simple grayscale conversion: average RGB or take first channel
            if (channels >= 3) {
                gray_data[i] = (unsigned char)((image_data[i * channels] + 
                                               image_data[i * channels + 1] + 
                                               image_data[i * channels + 2]) / 3);
            } else {
                gray_data[i] = image_data[i * channels];
            }
        }
        stbi_image_free(image_data);
        image_data = gray_data;
        channels = 1;
    }

    std::cout << "Loaded image: " << input_file << std::endl;
    std::cout << "Dimensions: " << width << "x" << height << " with " << C << " channel";
    if (C > 1) std::cout << "s";
    std::cout << " (RGB)" << std::endl;

    // Convert to float arrays for each channel
    float **h_input = new float*[C];
    for (int c = 0; c < C; c++) {
        h_input[c] = new float[M * N];
        for (int i = 0; i < M * N; i++) {
            h_input[c][i] = (float)image_data[i * C + c];
        }
    }

    // Host arrays for compressed representation (for each channel)
    float **h_R = new float*[C];      // Row sums
    float **h_C = new float*[C];      // Column sums  
    float **h_R_sq = new float*[C];   // Row sums of squares
    float **h_C_sq = new float*[C];   // Column sums of squares
    
    for (int c = 0; c < C; c++) {
        h_R[c] = new float[M];
        h_C[c] = new float[N];
        h_R_sq[c] = new float[M];
        h_C_sq[c] = new float[N];
    }

    // Device pointers
    float *d_input_float, *d_S_row, *d_S_col, *d_S_row_sq, *d_S_col_sq;
    cudaMalloc(&d_input_float, M * N * sizeof(float));
    cudaMalloc(&d_S_row, M * sizeof(float));
    cudaMalloc(&d_S_col, N * sizeof(float));
    cudaMalloc(&d_S_row_sq, M * sizeof(float));
    cudaMalloc(&d_S_col_sq, N * sizeof(float));

    // Process each channel
    int threads_per_block = min(256, max(N, M));
    int shared_mem_size = 2 * threads_per_block * sizeof(float);
    
    for (int c = 0; c < C; c++) {
        std::cout << "\nProcessing channel " << (c + 1) << "/" << C << "..." << std::endl;
        
        // Copy input channel to device
        cudaMemcpy(d_input_float, h_input[c], M * N * sizeof(float), cudaMemcpyHostToDevice);

        // Compute compressed representation using optimized kernels
        compute_row_stats<<<M, threads_per_block, shared_mem_size>>>(d_input_float, d_S_row, d_S_row_sq, M, N);
        compute_col_stats<<<N, threads_per_block, shared_mem_size>>>(d_input_float, d_S_col, d_S_col_sq, M, N);
        
        // Copy results back to host
        cudaMemcpy(h_R[c], d_S_row, M * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_R_sq[c], d_S_row_sq, M * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_C[c], d_S_col, N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_C_sq[c], d_S_col_sq, N * sizeof(float), cudaMemcpyDeviceToHost);
    }

    // Display compressed representation for small images
    if (M <= 20 && N <= 20 && C == 1) {
        std::cout << "\nCompressed Representation (4x" << max(M, N) << " matrix):\n";
        std::cout << "Row 1: Row Sums, Row 2: Column Sums, Row 3: Row Sums of Squares, Row 4: Column Sums of Squares\n";
        std::cout << std::string(max(M, N) * 8 + 2, '=') << "\n";
        
        // Display the 4 arrays as requested
        std::cout << " ";
        for (int i = 0; i < M; i++) printf("%7.1f ", h_R[0][i]);
        for (int i = M; i < max(M, N); i++) printf("%7.1f ", 0.0f);
        std::cout << "\n ";
        for (int j = 0; j < N; j++) printf("%7.1f ", h_C[0][j]);
        for (int j = N; j < max(M, N); j++) printf("%7.1f ", 0.0f);
        std::cout << "\n ";
        for (int i = 0; i < M; i++) printf("%7.0f ", h_R_sq[0][i]);
        for (int i = M; i < max(M, N); i++) printf("%7.0f ", 0.0f);
        std::cout << "\n ";
        for (int j = 0; j < N; j++) printf("%7.0f ", h_C_sq[0][j]);
        for (int j = N; j < max(M, N); j++) printf("%7.0f ", 0.0f);
        std::cout << "\n" << std::string(max(M, N) * 8 + 2, '=') << "\n";
    } else if (C > 1) {
        std::cout << "\nRGB image - compressed representation contains " << C << " sets of 4 arrays each.\n";
    } else {
        std::cout << "\nImage too large to display compressed representation matrix.\n";
    }

    // Save compressed representation to output file
    std::ofstream outfile(output_file, std::ios::binary);
    if (!outfile) {
        std::cerr << "Error: Cannot create output file " << output_file << std::endl;
        return 1;
    }

    // Write header: dimensions and channels
    outfile.write(reinterpret_cast<const char*>(&M), sizeof(int));
    outfile.write(reinterpret_cast<const char*>(&N), sizeof(int));
    outfile.write(reinterpret_cast<const char*>(&C), sizeof(int));
    
    // Write compressed data for each channel
    for (int c = 0; c < C; c++) {
        outfile.write(reinterpret_cast<const char*>(h_R[c]), M * sizeof(float));
        outfile.write(reinterpret_cast<const char*>(h_C[c]), N * sizeof(float));
        outfile.write(reinterpret_cast<const char*>(h_R_sq[c]), M * sizeof(float));
        outfile.write(reinterpret_cast<const char*>(h_C_sq[c]), N * sizeof(float));
    }
    
    outfile.close();

    // Calculate compression ratio
    size_t original_size = M * N * C * sizeof(unsigned char);
    size_t compressed_size = 3 * sizeof(int) + C * (2 * M + 2 * N) * sizeof(float);
    float compression_ratio = (float)original_size / compressed_size;

    std::cout << "\nCompression Statistics:\n";
    std::cout << "======================\n";
    std::cout << "Original size: " << original_size << " bytes\n";
    std::cout << "Compressed size: " << compressed_size << " bytes\n";
    std::cout << "Compression ratio: " << compression_ratio << ":1\n";
    std::cout << "Space savings: " << (1.0f - 1.0f/compression_ratio) * 100.0f << "%\n";
    
    std::cout << "\nCompression completed successfully!\n";
    std::cout << "Compressed data saved to: " << output_file << std::endl;

    // Free memory
    stbi_image_free(image_data);
    cudaFree(d_input_float);
    cudaFree(d_S_row);
    cudaFree(d_S_col);
    cudaFree(d_S_row_sq);
    cudaFree(d_S_col_sq);
    
    for (int c = 0; c < C; c++) {
        delete[] h_input[c];
        delete[] h_R[c];
        delete[] h_C[c];
        delete[] h_R_sq[c];
        delete[] h_C_sq[c];
    }
    delete[] h_input;
    delete[] h_R;
    delete[] h_C;
    delete[] h_R_sq;
    delete[] h_C_sq;

    return 0;
}
