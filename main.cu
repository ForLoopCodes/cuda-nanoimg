#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

// Kernel to compute row sums and sum of squares
__global__ void compute_row_stats(float *X, float *S_row, float *S_row_sq, int M, int N) {
    int row = blockIdx.x;
    if (row < M) {
        float sum = 0.0f;
        float sum_sq = 0.0f;
        
        for (int col = 0; col < N; col++) {
            float val = X[row * N + col];
            sum += val;
            sum_sq += val * val;
        }
        
        S_row[row] = sum;
        S_row_sq[row] = sum_sq;
    }
}

// Kernel to compute column sums and sum of squares
__global__ void compute_col_stats(float *X, float *S_col, float *S_col_sq, int M, int N) {
    int col = blockIdx.x;
    if (col < N) {
        float sum = 0.0f;
        float sum_sq = 0.0f;
        
        for (int row = 0; row < M; row++) {
            float val = X[row * N + col];
            sum += val;
            sum_sq += val * val;
        }
        
        S_col[col] = sum;
        S_col_sq[col] = sum_sq;
    }
}

// Kernel to scale rows to match target row sums and sums of squares
__global__ void scale_rows(float *X, float *S_row, float *R, float *S_row_sq, float *R_sq, int M, int N) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    if (row < M && col < N) {
        float k_sum = (S_row[row] > 0) ? R[row] / S_row[row] : 1.0f;
        float k_sq = (S_row_sq[row] > 0) ? sqrtf(R_sq[row] / S_row_sq[row]) : 1.0f;
        float k = (k_sum + k_sq) / 2.0f; // Average scaling factors
        X[row * N + col] *= k;
    }
}

// Kernel to scale columns to match target column sums and sums of squares
__global__ void scale_cols(float *X, float *S_col, float *C, float *S_col_sq, float *C_sq, int M, int N) {
    int col = blockIdx.x;
    int row = threadIdx.x;
    if (col < N && row < M) {
        float k_sum = (S_col[col] > 0) ? C[col] / S_col[col] : 1.0f;
        float k_sq = (S_col_sq[col] > 0) ? sqrtf(C_sq[col] / S_col_sq[col]) : 1.0f;
        float k = (k_sum + k_sq) / 2.0f; // Average scaling factors
        X[row * N + col] *= k;
    }
}

// Kernel to round float matrix to integers [0, 255]
__global__ void round_to_int(float *X_float, int *X_int, int M, int N) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    if (row < M && col < N) {
        int idx = row * N + col;
        float val = X_float[idx];
        int rounded = roundf(val);
        X_int[idx] = max(0, min(255, rounded)); // Clamp to [0, 255]
    }
}

// Kernel to convert integer matrix back to float
__global__ void convert_int_to_float(int *X_int, float *X_float, int M, int N) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    if (row < M && col < N) {
        int idx = row * N + col;
        X_float[idx] = (float)X_int[idx];
    }
}

int main() {
    const int M = 6; // Number of rows
    const int N = 6; // Number of columns

    // Input matrix (6x6, integers [0, 255])
    int h_input[6][6] = {
        {10, 20, 30, 40, 50, 60},
        {70, 80, 90, 100, 110, 120},
        {130, 140, 150, 160, 170, 180},
        {190, 200, 210, 220, 230, 240},
        {15, 25, 35, 45, 55, 65},
        {75, 85, 95, 105, 115, 125}
    };

    // Host arrays
    float h_R[6], h_C[6], h_R_sq[6], h_C_sq[6]; // Target sums
    float h_R_out[6], h_C_out[6], h_R_sq_out[6], h_C_sq_out[6]; // Output sums for verification
    float h_output_float[36];
    float h_input_float[36];
    int h_output_int[36];    // Convert input to float and initialize output
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            h_input_float[i * N + j] = (float)h_input[i][j];
            h_output_float[i * N + j] = 1.0f; // Initialize output to 1s
        }
    }    // Device pointers
    float *d_input_float, *d_output_float, *d_S_row, *d_S_col, *d_S_row_sq, *d_S_col_sq, *d_R, *d_C, *d_R_sq, *d_C_sq;
    int *d_output_int;
    cudaMalloc(&d_input_float, M * N * sizeof(float));
    cudaMalloc(&d_output_float, M * N * sizeof(float));
    cudaMalloc(&d_output_int, M * N * sizeof(int));
    cudaMalloc(&d_S_row, M * sizeof(float));
    cudaMalloc(&d_S_col, N * sizeof(float));
    cudaMalloc(&d_S_row_sq, M * sizeof(float));
    cudaMalloc(&d_S_col_sq, N * sizeof(float));
    cudaMalloc(&d_R, M * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));
    cudaMalloc(&d_R_sq, M * sizeof(float));
    cudaMalloc(&d_C_sq, N * sizeof(float));    // Copy input to device
    cudaMemcpy(d_input_float, h_input_float, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_float, h_output_float, M * N * sizeof(float), cudaMemcpyHostToDevice);    // Compute target sums
    compute_row_stats<<<M, 1>>>(d_input_float, d_S_row, d_S_row_sq, M, N);
    compute_col_stats<<<N, 1>>>(d_input_float, d_S_col, d_S_col_sq, M, N);
    cudaMemcpy(h_R, d_S_row, M * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_R_sq, d_S_row_sq, M * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C, d_S_col, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_sq, d_S_col_sq, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Copy target sums to device
    cudaMemcpy(d_R, h_R, M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_R_sq, h_R_sq, M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_sq, h_C_sq, N * sizeof(float), cudaMemcpyHostToDevice);    // Iterative scaling (100 iterations)
    for (int iter = 0; iter < 100; iter++) {
        compute_row_stats<<<M, 1>>>(d_output_float, d_S_row, d_S_row_sq, M, N);
        scale_rows<<<M, N>>>(d_output_float, d_S_row, d_R, d_S_row_sq, d_R_sq, M, N);
        compute_col_stats<<<N, 1>>>(d_output_float, d_S_col, d_S_col_sq, M, N);
        scale_cols<<<N, M>>>(d_output_float, d_S_col, d_C, d_S_col_sq, d_C_sq, M, N);
        round_to_int<<<M, N>>>(d_output_float, d_output_int, M, N);
        
        // Convert integer output back to float for next iteration
        int* d_temp_int = d_output_int;
        float* d_temp_float = d_output_float;
        for (int i = 0; i < M * N; i++) {
            // This needs to be done with a kernel, not with cudaMemcpy
        }
        
        // Use a kernel to convert int to float
        convert_int_to_float<<<M, N>>>(d_output_int, d_output_float, M, N);
    }    // Compute sums for reconstructed matrix
    compute_row_stats<<<M, 1>>>(d_output_float, d_S_row, d_S_row_sq, M, N);
    compute_col_stats<<<N, 1>>>(d_output_float, d_S_col, d_S_col_sq, M, N);
    cudaMemcpy(h_R_out, d_S_row, M * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_R_sq_out, d_S_row_sq, M * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_out, d_S_col, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_sq_out, d_S_col_sq, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Copy final reconstructed matrix to host
    cudaMemcpy(h_output_int, d_output_int, M * N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print input matrix
    std::cout << "Input Matrix:\n";
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << h_input[i][j] << " ";
        }
        std::cout << std::endl;
    }

    // Print target sums
    std::cout << "\nTarget Row Sums:\n";
    for (int i = 0; i < M; i++) {
        std::cout << h_R[i] << " ";
    }
    std::cout << "\nTarget Column Sums:\n";
    for (int j = 0; j < N; j++) {
        std::cout << h_C[j] << " ";
    }
    std::cout << "\nTarget Row Sums of Squares:\n";
    for (int i = 0; i < M; i++) {
        std::cout << h_R_sq[i] << " ";
    }
    std::cout << "\nTarget Column Sums of Squares:\n";
    for (int j = 0; j < N; j++) {
        std::cout << h_C_sq[j] << " ";
    }

    // Print reconstructed matrix
    std::cout << "\nReconstructed Matrix (integers [0, 255]):\n";
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << h_output_int[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    // Verify reconstructed matrix
    std::cout << "\nVerification:\n";
    float tolerance = 1e-2; // Small tolerance for floating-point errors
    bool verified = true;
    for (int i = 0; i < M; i++) {
        if (fabs(h_R_out[i] - h_R[i]) > tolerance * h_R[i]) {
            std::cout << "Row sum " << i << " mismatch: Expected " << h_R[i] << ", Got " << h_R_out[i] << "\n";
            verified = false;
        }
        if (fabs(h_R_sq_out[i] - h_R_sq[i]) > tolerance * h_R_sq[i]) {
            std::cout << "Row sum of squares " << i << " mismatch: Expected " << h_R_sq[i] << ", Got " << h_R_sq_out[i] << "\n";
            verified = false;
        }
    }
    for (int j = 0; j < N; j++) {
        if (fabs(h_C_out[j] - h_C[j]) > tolerance * h_C[j]) {
            std::cout << "Column sum " << j << " mismatch: Expected " << h_C[j] << ", Got " << h_C_out[j] << "\n";
            verified = false;
        }
        if (fabs(h_C_sq_out[j] - h_C_sq[j]) > tolerance * h_C_sq[j]) {
            std::cout << "Column sum of squares " << j << " mismatch: Expected " << h_C_sq[j] << ", Got " << h_C_sq_out[j] << "\n";
            verified = false;
        }
    }
    if (verified) {
        std::cout << "Reconstructed matrix satisfies all constraints within tolerance.\n";
    } else {
        std::cout << "Reconstructed matrix does not fully satisfy constraints.\n";
    }    // Free device memory
    cudaFree(d_input_float);
    cudaFree(d_output_float);
    cudaFree(d_output_int);
    cudaFree(d_S_row);
    cudaFree(d_S_col);
    cudaFree(d_S_row_sq);
    cudaFree(d_S_col_sq);
    cudaFree(d_R);
    cudaFree(d_C);
    cudaFree(d_R_sq);
    cudaFree(d_C_sq);

    return 0;
}