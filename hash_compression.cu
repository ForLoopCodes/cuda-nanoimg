#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <time.h>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

// Hash function types
#define HASH_SIMPLE_SUM     0
#define HASH_MODULAR_SUM    1
#define HASH_POLYNOMIAL     2
#define HASH_FNV1A          3

// Compression methods
#define METHOD_COLUMN_HASH  0
#define METHOD_BLOCK_HASH   1

// Structure to hold compressed data
struct CompressedData {
    int width, height;
    int method;
    int hash_type;
    int block_size;  // For block-based method
    int num_hashes;
    uint32_t* hashes;
    
    // Constructor
    CompressedData(int w, int h, int m, int ht, int bs = 1) : 
        width(w), height(h), method(m), hash_type(ht), block_size(bs) {
        if (method == METHOD_COLUMN_HASH) {
            num_hashes = width;
        } else {
            int blocks_x = (width + block_size - 1) / block_size;
            int blocks_y = (height + block_size - 1) / block_size;
            num_hashes = blocks_x * blocks_y;
        }
        hashes = new uint32_t[num_hashes];
    }
    
    ~CompressedData() {
        delete[] hashes;
    }
    
    size_t get_compressed_size() const {
        return sizeof(int) * 5 + sizeof(uint32_t) * num_hashes;
    }
    
    float get_compression_ratio(int original_pixels) const {
        return (float)original_pixels / (get_compressed_size() / sizeof(uint8_t));
    }
};

// CUDA kernel to compute column hashes using simple sum
__global__ void compute_column_hash_simple(uint8_t* image, uint32_t* hashes, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (col < width) {
        uint32_t sum = 0;
        for (int row = 0; row < height; row++) {
            sum += image[row * width + col];
        }
        hashes[col] = sum;
    }
}

// CUDA kernel to compute column hashes using modular arithmetic (sum mod prime)
__global__ void compute_column_hash_modular(uint8_t* image, uint32_t* hashes, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t PRIME = 65521; // Large prime for better distribution
    
    if (col < width) {
        uint32_t sum = 0;
        for (int row = 0; row < height; row++) {
            sum = (sum + image[row * width + col]) % PRIME;
        }
        hashes[col] = sum;
    }
}

// CUDA kernel to compute column hashes using polynomial hash
__global__ void compute_column_hash_polynomial(uint8_t* image, uint32_t* hashes, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t BASE = 257;
    const uint32_t MOD = 1000000007;
    
    if (col < width) {
        uint64_t hash_val = 0;
        uint64_t power = 1;
        
        for (int row = 0; row < height; row++) {
            hash_val = (hash_val + ((uint64_t)image[row * width + col] * power)) % MOD;
            power = (power * BASE) % MOD;
        }
        hashes[col] = (uint32_t)hash_val;
    }
}

// CUDA kernel to compute column hashes using FNV-1a hash
__global__ void compute_column_hash_fnv1a(uint8_t* image, uint32_t* hashes, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t FNV_PRIME = 16777619u;
    const uint32_t FNV_OFFSET = 2166136261u;
    
    if (col < width) {
        uint32_t hash_val = FNV_OFFSET;
        
        for (int row = 0; row < height; row++) {
            hash_val ^= image[row * width + col];
            hash_val *= FNV_PRIME;
        }
        hashes[col] = hash_val;
    }
}

// CUDA kernel to compute block hashes (10x10 or other sizes)
__global__ void compute_block_hash_modular(uint8_t* image, uint32_t* hashes, 
                                          int width, int height, int block_size) {
    int block_x = blockIdx.x * blockDim.x + threadIdx.x;
    int block_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    int blocks_x = (width + block_size - 1) / block_size;
    int blocks_y = (height + block_size - 1) / block_size;
    
    if (block_x < blocks_x && block_y < blocks_y) {
        const uint32_t PRIME = 65521;
        uint32_t sum = 0;
        
        int start_x = block_x * block_size;
        int start_y = block_y * block_size;
        int end_x = min(start_x + block_size, width);
        int end_y = min(start_y + block_size, height);
        
        for (int y = start_y; y < end_y; y++) {
            for (int x = start_x; x < end_x; x++) {
                sum = (sum + image[y * width + x]) % PRIME;
            }
        }
        
        hashes[block_y * blocks_x + block_x] = sum;
    }
}

// CUDA kernel for reconstruction - iterative approach for column hashes
__global__ void reconstruct_column_iterative(uint8_t* image, uint32_t* target_hashes, 
                                            int width, int height, int hash_type, 
                                            curandState* states, int iteration) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (col < width) {
        curandState localState = states[col];
        
        // Compute current hash for this column
        uint32_t current_hash = 0;
        const uint32_t PRIME = 65521;
        
        if (hash_type == HASH_SIMPLE_SUM) {
            for (int row = 0; row < height; row++) {
                current_hash += image[row * width + col];
            }
        } else if (hash_type == HASH_MODULAR_SUM) {
            for (int row = 0; row < height; row++) {
                current_hash = (current_hash + image[row * width + col]) % PRIME;
            }
        }
        
        // If hash doesn't match, modify a random pixel in this column
        if (current_hash != target_hashes[col]) {
            int random_row = curand(&localState) % height;
            int pixel_idx = random_row * width + col;
            
            // Try to adjust pixel value to get closer to target hash
            uint8_t current_val = image[pixel_idx];
            int hash_diff = (int)target_hashes[col] - (int)current_hash;
            
            if (hash_type == HASH_SIMPLE_SUM) {
                // For simple sum, we can directly adjust
                int new_val = (int)current_val + hash_diff;
                new_val = max(0, min(255, new_val));
                image[pixel_idx] = (uint8_t)new_val;
            } else {
                // For modular hash, try small random adjustments
                int adjustment = (curand(&localState) % 21) - 10; // -10 to +10
                int new_val = (int)current_val + adjustment;
                new_val = max(0, min(255, new_val));
                image[pixel_idx] = (uint8_t)new_val;
            }
        }
        
        states[col] = localState;
    }
}

// CUDA kernel to initialize random states
__global__ void init_random_states(curandState* states, int n, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// Host function to compress image using column hashing
CompressedData* compress_image_column_hash(uint8_t* image, int width, int height, int hash_type) {
    CompressedData* compressed = new CompressedData(width, height, METHOD_COLUMN_HASH, hash_type);
    
    // Allocate device memory
    uint8_t* d_image;
    uint32_t* d_hashes;
    
    cudaMalloc(&d_image, width * height * sizeof(uint8_t));
    cudaMalloc(&d_hashes, width * sizeof(uint32_t));
    
    // Copy image to device
    cudaMemcpy(d_image, image, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice);
    
    // Compute hashes
    int threads_per_block = 256;
    int blocks = (width + threads_per_block - 1) / threads_per_block;
    
    switch (hash_type) {
        case HASH_SIMPLE_SUM:
            compute_column_hash_simple<<<blocks, threads_per_block>>>(d_image, d_hashes, width, height);
            break;
        case HASH_MODULAR_SUM:
            compute_column_hash_modular<<<blocks, threads_per_block>>>(d_image, d_hashes, width, height);
            break;
        case HASH_POLYNOMIAL:
            compute_column_hash_polynomial<<<blocks, threads_per_block>>>(d_image, d_hashes, width, height);
            break;
        case HASH_FNV1A:
            compute_column_hash_fnv1a<<<blocks, threads_per_block>>>(d_image, d_hashes, width, height);
            break;
    }
    
    // Copy hashes back to host
    cudaMemcpy(compressed->hashes, d_hashes, width * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
    // Clean up device memory
    cudaFree(d_image);
    cudaFree(d_hashes);
    
    return compressed;
}

// Host function to compress image using block hashing
CompressedData* compress_image_block_hash(uint8_t* image, int width, int height, int block_size, int hash_type) {
    CompressedData* compressed = new CompressedData(width, height, METHOD_BLOCK_HASH, hash_type, block_size);
    
    // Allocate device memory
    uint8_t* d_image;
    uint32_t* d_hashes;
    
    int blocks_x = (width + block_size - 1) / block_size;
    int blocks_y = (height + block_size - 1) / block_size;
    int total_blocks = blocks_x * blocks_y;
    
    cudaMalloc(&d_image, width * height * sizeof(uint8_t));
    cudaMalloc(&d_hashes, total_blocks * sizeof(uint32_t));
    
    // Copy image to device
    cudaMemcpy(d_image, image, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice);
    
    // Compute block hashes
    dim3 threads_per_block(16, 16);
    dim3 grid_blocks((blocks_x + threads_per_block.x - 1) / threads_per_block.x,
                     (blocks_y + threads_per_block.y - 1) / threads_per_block.y);
    
    // Currently implementing only modular hash for blocks
    compute_block_hash_modular<<<grid_blocks, threads_per_block>>>(d_image, d_hashes, width, height, block_size);
    
    // Copy hashes back to host
    cudaMemcpy(compressed->hashes, d_hashes, total_blocks * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
    // Clean up device memory
    cudaFree(d_image);
    cudaFree(d_hashes);
    
    return compressed;
}

// Host function to reconstruct image from column hashes
uint8_t* reconstruct_from_column_hash(CompressedData* compressed, int max_iterations = 10000) {
    int width = compressed->width;
    int height = compressed->height;
    
    // Allocate host and device memory
    uint8_t* h_image = new uint8_t[width * height];
    uint8_t* d_image;
    uint32_t* d_hashes;
    curandState* d_states;
    
    cudaMalloc(&d_image, width * height * sizeof(uint8_t));
    cudaMalloc(&d_hashes, width * sizeof(uint32_t));
    cudaMalloc(&d_states, width * sizeof(curandState));
    
    // Initialize with random values
    srand(time(NULL));
    for (int i = 0; i < width * height; i++) {
        h_image[i] = rand() % 256;
    }
    
    // Copy initial image and target hashes to device
    cudaMemcpy(d_image, h_image, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hashes, compressed->hashes, width * sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    // Initialize random states
    int threads_per_block = 256;
    int blocks = (width + threads_per_block - 1) / threads_per_block;
    init_random_states<<<blocks, threads_per_block>>>(d_states, width, time(NULL));
    
    // Iterative reconstruction
    for (int iter = 0; iter < max_iterations; iter++) {
        reconstruct_column_iterative<<<blocks, threads_per_block>>>(
            d_image, d_hashes, width, height, compressed->hash_type, d_states, iter);
        
        // Check convergence every 1000 iterations
        if (iter % 1000 == 999) {
            // Copy current image back and check hash matches
            cudaMemcpy(h_image, d_image, width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost);
            
            int matches = 0;
            for (int col = 0; col < width; col++) {
                uint32_t current_hash = 0;
                
                if (compressed->hash_type == HASH_SIMPLE_SUM) {
                    for (int row = 0; row < height; row++) {
                        current_hash += h_image[row * width + col];
                    }
                } else if (compressed->hash_type == HASH_MODULAR_SUM) {
                    const uint32_t PRIME = 65521;
                    for (int row = 0; row < height; row++) {
                        current_hash = (current_hash + h_image[row * width + col]) % PRIME;
                    }
                }
                
                if (current_hash == compressed->hashes[col]) {
                    matches++;
                }
            }
            
            float match_rate = (float)matches / width;
            printf("Iteration %d: %.1f%% hash matches\n", iter + 1, match_rate * 100.0f);
            
            if (match_rate > 0.95f) {  // 95% match threshold
                printf("Convergence achieved at iteration %d\n", iter + 1);
                break;
            }
        }
    }
    
    // Copy final result back to host
    cudaMemcpy(h_image, d_image, width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    
    // Clean up device memory
    cudaFree(d_image);
    cudaFree(d_hashes);
    cudaFree(d_states);
    
    return h_image;
}

// Function to analyze compression performance
void analyze_compression(const char* image_path) {
    printf("=== Hash-Based Image Compression Analysis ===\n");
    printf("Loading image: %s\n", image_path);
    
    // Load image
    int width, height, channels;
    uint8_t* image = stbi_load(image_path, &width, &height, &channels, 1); // Force grayscale
    
    if (!image) {
        printf("Failed to load image: %s\n", image_path);
        return;
    }
    
    printf("Image dimensions: %dx%d (%d pixels)\n", width, height, width * height);
    printf("Original size: %d bytes\n", width * height);
    
    // Test different compression methods
    const char* hash_names[] = {"Simple Sum", "Modular Sum", "Polynomial", "FNV-1a"};
    
    printf("\n--- Column-Based Hashing ---\n");
    for (int hash_type = 0; hash_type < 4; hash_type++) {
        printf("\nTesting %s hash:\n", hash_names[hash_type]);
        
        CompressedData* compressed = compress_image_column_hash(image, width, height, hash_type);
        
        printf("  Compressed size: %zu bytes\n", compressed->get_compressed_size());
        printf("  Compression ratio: %.1f:1\n", compressed->get_compression_ratio(width * height));
        printf("  Number of hashes: %d\n", compressed->num_hashes);
        
        // Test reconstruction for simple hash types only (faster)
        if (hash_type <= HASH_MODULAR_SUM) {
            printf("  Testing reconstruction...\n");
            uint8_t* reconstructed = reconstruct_from_column_hash(compressed, 5000);
            
            // Calculate reconstruction error
            float total_error = 0.0f;
            for (int i = 0; i < width * height; i++) {
                total_error += abs((int)image[i] - (int)reconstructed[i]);
            }
            float avg_error = total_error / (width * height);
            float accuracy = 100.0f * (1.0f - avg_error / 255.0f);
            
            printf("  Average reconstruction error: %.2f\n", avg_error);
            printf("  Reconstruction accuracy: %.1f%%\n", accuracy);
            
            // Save reconstructed image
            char output_name[256];
            sprintf(output_name, "reconstructed_%s_hash.jpg", 
                   hash_type == HASH_SIMPLE_SUM ? "simple" : "modular");
            stbi_write_jpg(output_name, width, height, 1, reconstructed, 90);
            printf("  Saved reconstructed image: %s\n", output_name);
            
            delete[] reconstructed;
        }
        
        delete compressed;
    }
    
    printf("\n--- Block-Based Hashing (10x10) ---\n");
    CompressedData* block_compressed = compress_image_block_hash(image, width, height, 10, HASH_MODULAR_SUM);
    
    printf("Compressed size: %zu bytes\n", block_compressed->get_compressed_size());
    printf("Compression ratio: %.1f:1\n", block_compressed->get_compression_ratio(width * height));
    printf("Number of blocks: %d\n", block_compressed->num_hashes);
    
    // Calculate theoretical compression ratios for different image sizes
    printf("\n--- Theoretical Analysis ---\n");
    int test_sizes[][2] = {{6, 6}, {100, 100}, {500, 500}, {1000, 1000}, {2000, 2000}};
    
    for (int i = 0; i < 5; i++) {
        int w = test_sizes[i][0];
        int h = test_sizes[i][1];
        int pixels = w * h;
        
        // Column hashing: w hashes + 2 dimension integers
        int col_compressed_size = w * sizeof(uint32_t) + 2 * sizeof(int);
        float col_ratio = (float)pixels / col_compressed_size;
        
        // Block hashing (10x10): number of blocks * hash size + metadata
        int blocks = ((w + 9) / 10) * ((h + 9) / 10);
        int block_compressed_size = blocks * sizeof(uint32_t) + 5 * sizeof(int);
        float block_ratio = (float)pixels / block_compressed_size;
        
        printf("%dx%d image:\n", w, h);
        printf("  Column hash compression: %.1f:1\n", col_ratio);
        printf("  Block hash compression: %.1f:1\n", block_ratio);
    }
    
    delete block_compressed;
    stbi_image_free(image);
}

int main(int argc, char** argv) {
    // Check CUDA device
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        printf("No CUDA devices found!\n");
        return -1;
    }
    
    printf("CUDA devices found: %d\n", device_count);
    
    // Use test image or provided image
    const char* image_path = "test_in.jpg";
    if (argc > 1) {
        image_path = argv[1];
    }
    
    analyze_compression(image_path);
    
    return 0;
}
