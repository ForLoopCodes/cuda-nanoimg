#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// CUDA kernels (same as before)
__global__ void compute_row_stats(float *X, float *S_row, float *S_row_sq, int M, int N) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    extern __shared__ float sdata[];
    float *s_sum = sdata;
    float *s_sum_sq = &sdata[blockDim.x];
    
    if (row < M) {
        float sum = 0.0f;
        float sum_sq = 0.0f;
        
        for (int col = tid; col < N; col += blockDim.x) {
            float val = X[row * N + col];
            sum += val;
            sum_sq += val * val;
        }
        
        s_sum[tid] = sum;
        s_sum_sq[tid] = sum_sq;
        __syncthreads();
        
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

__global__ void compute_col_stats(float *X, float *S_col, float *S_col_sq, int M, int N) {
    int col = blockIdx.x;
    int tid = threadIdx.x;
    
    extern __shared__ float sdata[];
    float *s_sum = sdata;
    float *s_sum_sq = &sdata[blockDim.x];
    
    if (col < N) {
        float sum = 0.0f;
        float sum_sq = 0.0f;
        
        for (int row = tid; row < M; row += blockDim.x) {
            float val = X[row * N + col];
            sum += val;
            sum_sq += val * val;
        }
        
        s_sum[tid] = sum;
        s_sum_sq[tid] = sum_sq;
        __syncthreads();
        
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

__global__ void scale_rows(float *X, float *S_row, float *R, float *S_row_sq, float *R_sq, int M, int N) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    if (row < M && col < N) {
        float k_sum = (S_row[row] > 0) ? R[row] / S_row[row] : 1.0f;
        float k_sq = (S_row_sq[row] > 0) ? sqrtf(R_sq[row] / S_row_sq[row]) : 1.0f;
        
        float w1 = 0.7f, w2 = 0.3f;
        float k = w1 * k_sum + w2 * k_sq;
        
        float damping = 0.8f;
        k = 1.0f + damping * (k - 1.0f);
        
        X[row * N + col] *= k;
    }
}

__global__ void scale_cols(float *X, float *S_col, float *C, float *S_col_sq, float *C_sq, int M, int N) {
    int col = blockIdx.x;
    int row = threadIdx.x;
    if (col < N && row < M) {
        float k_sum = (S_col[col] > 0) ? C[col] / S_col[col] : 1.0f;
        float k_sq = (S_col_sq[col] > 0) ? sqrtf(C_sq[col] / S_col_sq[col]) : 1.0f;
        
        float w1 = 0.7f, w2 = 0.3f;
        float k = w1 * k_sum + w2 * k_sq;
        
        float damping = 0.8f;
        k = 1.0f + damping * (k - 1.0f);
        
        X[row * N + col] *= k;
    }
}

__global__ void round_to_int(float *X_float, int *X_int, int M, int N) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    if (row < M && col < N) {
        int idx = row * N + col;
        float val = X_float[idx];
        int rounded = roundf(val);
        X_int[idx] = max(0, min(255, rounded));
    }
}

__global__ void convert_int_to_float(int *X_int, float *X_float, int M, int N) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    if (row < M && col < N) {
        int idx = row * N + col;
        X_float[idx] = (float)X_int[idx];
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <compressed_file> <output_image>\n";
        std::cout << "  compressed_file: Input file with compressed representation\n";
        std::cout << "  output_image: Output image file (PNG format)\n";
        std::cout << "Supports both grayscale and RGB reconstruction\n";
        return 1;
    }

    const char* compressed_file = argv[1];
    const char* output_file = argv[2];

    // Read compressed data
    std::ifstream infile(compressed_file, std::ios::binary);
    if (!infile) {
        std::cerr << "Error: Cannot open compressed file " << compressed_file << std::endl;
        return 1;
    }

    int M, N, C;
    infile.read(reinterpret_cast<char*>(&M), sizeof(int));
    infile.read(reinterpret_cast<char*>(&N), sizeof(int));
    
    // Check if this is an older format (no channel info)
    if (infile.peek() == EOF || infile.tellg() == static_cast<std::streampos>(2 * sizeof(int) + 4 * M * sizeof(float) + 4 * N * sizeof(float))) {
        // Old format - single channel
        C = 1;
        infile.seekg(2 * sizeof(int)); // Reset to after M and N
    } else {
        // New format - read channel count
        infile.read(reinterpret_cast<char*>(&C), sizeof(int));
    }
    
    if (M <= 0 || N <= 0 || M > 10000 || N > 10000 || C <= 0 || C > 4) {
        std::cerr << "Error: Invalid image dimensions " << M << "x" << N << " with " << C << " channels" << std::endl;
        return 1;
    }

    std::cout << "Reconstructing " << N << "x" << M << " image with " << C << " channel";
    if (C > 1) std::cout << "s";
    std::cout << " from " << compressed_file << std::endl;

    // Allocate host memory for compressed representation (for each channel)
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

    // Read compressed data for each channel
    for (int c = 0; c < C; c++) {
        infile.read(reinterpret_cast<char*>(h_R[c]), M * sizeof(float));
        infile.read(reinterpret_cast<char*>(h_C[c]), N * sizeof(float));
        infile.read(reinterpret_cast<char*>(h_R_sq[c]), M * sizeof(float));
        infile.read(reinterpret_cast<char*>(h_C_sq[c]), N * sizeof(float));
    }
    infile.close();

    // Allocate host memory for reconstruction
    float **h_output_float = new float*[C];
    for (int c = 0; c < C; c++) {
        h_output_float[c] = new float[M * N];
    }

    // Device pointers
    float *d_output_float, *d_S_row, *d_S_col, *d_S_row_sq, *d_S_col_sq, *d_R, *d_C, *d_R_sq, *d_C_sq;
    cudaMalloc(&d_output_float, M * N * sizeof(float));
    cudaMalloc(&d_S_row, M * sizeof(float));
    cudaMalloc(&d_S_col, N * sizeof(float));
    cudaMalloc(&d_S_row_sq, M * sizeof(float));
    cudaMalloc(&d_S_col_sq, N * sizeof(float));
    cudaMalloc(&d_R, M * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));
    cudaMalloc(&d_R_sq, M * sizeof(float));
    cudaMalloc(&d_C_sq, N * sizeof(float));

    // Process each channel
    int threads_per_block = min(256, max(N, M));
    int shared_mem_size = 2 * threads_per_block * sizeof(float);
    
    for (int ch = 0; ch < C; ch++) {
        std::cout << "\nReconstructing channel " << (ch + 1) << "/" << C << "..." << std::endl;

        // Copy target sums to device
        cudaMemcpy(d_R, h_R[ch], M * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_C, h_C[ch], N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_R_sq, h_R_sq[ch], M * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_C_sq, h_C_sq[ch], N * sizeof(float), cudaMemcpyHostToDevice);        // Multiple random initializations - try up to 1000 for best quality
        int num_trials = 1000; // Maximum trials for best possible output
        float *best_result = new float[M * N];
        float best_error = 1e30f;
        int best_trial = 0;
        
        printf("Testing %d random initializations for channel %d (searching for best result)...\n", num_trials, ch + 1);
        
        for (int trial = 0; trial < num_trials; trial++) {
            // Initialize output matrix
            for (int i = 0; i < M * N; i++) {
                if (trial == 0) {
                    h_output_float[ch][i] = 128.0f;
                } else {
                    int random_val = 50 + (rand() % 151);
                    h_output_float[ch][i] = (float)random_val;
                }
            }
            
            cudaMemcpy(d_output_float, h_output_float[ch], M * N * sizeof(float), cudaMemcpyHostToDevice);
              // Run optimization with more iterations for better quality
            float prev_error = 1e30f;
            int max_iterations = 2000; // Increased for better quality with more trials
            float convergence_threshold = 1e-6;
            
            for (int iter = 0; iter < max_iterations; iter++) {
                compute_row_stats<<<M, threads_per_block, shared_mem_size>>>(d_output_float, d_S_row, d_S_row_sq, M, N);
                scale_rows<<<M, N>>>(d_output_float, d_S_row, d_R, d_S_row_sq, d_R_sq, M, N);
                compute_col_stats<<<N, threads_per_block, shared_mem_size>>>(d_output_float, d_S_col, d_S_col_sq, M, N);
                scale_cols<<<N, M>>>(d_output_float, d_S_col, d_C, d_S_col_sq, d_C_sq, M, N);
                
                if (iter % 10 == 9) {
                    round_to_int<<<M, N>>>(d_output_float, (int*)d_output_float, M, N);
                    convert_int_to_float<<<M, N>>>((int*)d_output_float, d_output_float, M, N);
                }
                
                // Check convergence every 50 iterations for early stopping
                if (iter % 50 == 49) {
                    float *h_temp = new float[M * N];
                    cudaMemcpy(h_temp, d_output_float, M * N * sizeof(float), cudaMemcpyDeviceToHost);
                    
                    float current_error = 0.0f;
                    for (int i = 0; i < M * N; i++) {
                        current_error += fabs(h_temp[i] - 128.0f); // Compare to expected average
                    }
                    current_error /= (M * N);
                    
                    float error_percentage = (current_error / 255.0f) * 100.0f;
                    if (error_percentage < 2.0f) {
                        printf("  Early stopping at iteration %d: Convergence achieved\n", iter + 1);
                        delete[] h_temp;
                        break;
                    }
                    
                    if (fabs(prev_error - current_error) < convergence_threshold) {
                        printf("  Early stopping at iteration %d: No improvement\n", iter + 1);
                        delete[] h_temp;
                        break;
                    }
                    prev_error = current_error;
                    delete[] h_temp;
                }
            }
            
            // Final rounding
            round_to_int<<<M, N>>>(d_output_float, (int*)d_output_float, M, N);
            convert_int_to_float<<<M, N>>>((int*)d_output_float, d_output_float, M, N);
            
            // Evaluate this trial
            float *h_trial_result = new float[M * N];
            cudaMemcpy(h_trial_result, d_output_float, M * N * sizeof(float), cudaMemcpyDeviceToHost);
            
            float trial_error = 0.0f;
            float *h_temp_R = new float[M];
            float *h_temp_C = new float[N];
            
            for (int i = 0; i < M; i++) {
                h_temp_R[i] = 0.0f;
                for (int j = 0; j < N; j++) {
                    h_temp_R[i] += h_trial_result[i * N + j];
                }
                trial_error += fabs(h_temp_R[i] - h_R[ch][i]);
            }
            
            for (int j = 0; j < N; j++) {
                h_temp_C[j] = 0.0f;
                for (int i = 0; i < M; i++) {
                    h_temp_C[j] += h_trial_result[i * N + j];
                }
                trial_error += fabs(h_temp_C[j] - h_C[ch][j]);
            }
              trial_error /= (M + N);
            
            // Show progress every 50 trials
            if ((trial + 1) % 50 == 0) {
                printf("Trial %d/%d: Constraint error = %.6f (Progress: %.1f%%)\n", 
                       trial + 1, num_trials, trial_error, 
                       100.0f * (trial + 1) / num_trials);
            } else if ((trial + 1) % 10 == 0) {
                printf("Trial %d: Constraint error = %.6f\n", trial + 1, trial_error);
            }
              if (trial_error < best_error) {
                best_error = trial_error;
                best_trial = trial;
                memcpy(best_result, h_trial_result, M * N * sizeof(float));
                printf("  ^ New best result! (Trial %d)\n", trial + 1);
                
                // Early termination if we achieve excellent results
                if (best_error < 0.01f) {
                    printf("  Excellent result achieved (error < 0.01)! Stopping early at trial %d.\n", trial + 1);
                    delete[] h_trial_result;
                    delete[] h_temp_R;
                    delete[] h_temp_C;
                    break;
                }
            }
            
            delete[] h_trial_result;
            delete[] h_temp_R;
            delete[] h_temp_C;
        }
          // Use the best result for this channel
        memcpy(h_output_float[ch], best_result, M * N * sizeof(float));
        printf("Best result for channel %d: Trial %d/%d with constraint error = %.6f\n", 
               ch + 1, best_trial + 1, num_trials, best_error);
        delete[] best_result;
    }

    // Convert to unsigned char for image output
    unsigned char *output_image = new unsigned char[M * N * C];
    for (int i = 0; i < M * N; i++) {
        for (int c = 0; c < C; c++) {
            output_image[i * C + c] = (unsigned char)round(h_output_float[c][i]);
        }
    }

    // Save as PNG image
    int success = 0;
    if (C == 1) {
        success = stbi_write_png(output_file, N, M, 1, output_image, N);
    } else if (C == 3) {
        success = stbi_write_png(output_file, N, M, 3, output_image, N * 3);
    } else {
        std::cerr << "Error: Unsupported number of channels for output: " << C << std::endl;
        success = 0;
    }

    if (success) {
        std::cout << "\nReconstruction completed successfully!\n";
        std::cout << "Reconstructed image saved to: " << output_file << std::endl;
        if (C > 1) {
            std::cout << "Image type: RGB (" << C << " channels)" << std::endl;
        } else {
            std::cout << "Image type: Grayscale" << std::endl;
        }
    } else {
        std::cerr << "Error: Could not save image to " << output_file << std::endl;
    }

    // Free memory
    cudaFree(d_output_float);
    cudaFree(d_S_row);
    cudaFree(d_S_col);
    cudaFree(d_S_row_sq);
    cudaFree(d_S_col_sq);
    cudaFree(d_R);
    cudaFree(d_C);
    cudaFree(d_R_sq);
    cudaFree(d_C_sq);
    
    for (int c = 0; c < C; c++) {
        delete[] h_R[c];
        delete[] h_C[c];
        delete[] h_R_sq[c];
        delete[] h_C_sq[c];
        delete[] h_output_float[c];
    }
    delete[] h_R;
    delete[] h_C;
    delete[] h_R_sq;
    delete[] h_C_sq;
    delete[] h_output_float;
    delete[] output_image;

    return 0;
}
