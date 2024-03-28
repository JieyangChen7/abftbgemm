#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <chrono>
using namespace std::chrono;



#include "abft_encoder.cpp"
#include "abft_corrector.cu"
#include "abft_checker.cpp"
#include "opt_kernels.cpp"

// #define M 5
// #define N 5
// #define K 3
// #define NUM_BATCHES 1

// int64_t m = 5;
// int64_t n = 5;
// int64_t k = 3;
// int64_t num_batches = 1;



template <typename T>
void MaxtrixRandom(T *A, int64_t num_batches, int64_t stride, int64_t ld, int64_t row, int64_t col){
  for(int num = 0; num < num_batches; num++){
    for (int r = 0; r < row; r++){
      for (int c = 0; c < col; c++){
        A[num*stride + c*ld + r] = ((T)rand() / RAND_MAX);
        // (half)((T)(rand()) / (T)(rand()));
        // A[num*stride + c*ld + r] = 1;
      }
    }
  }
}

template <typename T>
void outputChk(T *A, int64_t nb, int64_t ld, int64_t stride, int64_t row, int64_t col){
  size_t size = nb * (row * col) * sizeof(T);
  T *tensor;
  tensor = (T *)malloc(size);
  cudaMemcpy(tensor, A, size, cudaMemcpyDeviceToHost);
  for(int i = 0; i < nb; i++){
    printf("[ \n");
    for(int r = 0; r < row; r++){
      for(int c = 0; c < col; c++){
        printf("%.6f", T(tensor[i*stride + c*ld + r]));
        printf(", ");
      }
      printf("\n");
    }
    printf("]\n");
  }
  free(tensor);
}

template <typename T, int M, int N, int K>
void abftbgemm(int64_t m, int64_t n, int64_t k, T alpha,
    T *dA, int64_t ldda, int64_t stridea, 
    T *dB, int64_t lddb, int64_t strideb, T beta,
    T *dC, int64_t lddc, int64_t stridec,
    T *dA_colchk, int64_t ldda_colchk, T *dA_rowchk, int64_t ldda_rowchk,
    T *dA_colchk_r, int64_t ldda_colchk_r, T *dA_rowchk_r, int64_t ldda_rowchk_r,
    T *dB_colchk, int64_t lddb_colchk, T *dB_rowchk, int64_t lddb_rowchk,    
    T *dB_colchk_r, int64_t lddb_colchk_r, T *dB_rowchk_r, int64_t lddb_rowchk_r,
    T *dC_colchk, int64_t lddc_colchk, T *dC_rowchk, int64_t lddc_rowchk,
    T *dC_colchk_r, int64_t lddc_colchk_r, T *dC_rowchk_r, int64_t lddc_rowchk_r,
    T *chk_v_a, T *chk_v_b, int64_t ld_chk_v,
    int64_t num_batches,
    bool COL_FT, bool ROW_FT, bool DEBUG, bool CHECK_BEFORE, bool CHECK_AFTER){
    
    std::cout << "Using abftbgemm-at::T function." << std::endl;

    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaStream_t stream_main, stream_colchk, stream_rowchk;
    cudaStreamCreate(&stream_main);
    cudaStreamCreate(&stream_colchk);
    cudaStreamCreate(&stream_rowchk);
    cublasSetStream(handle, stream_main);

    cudaEvent_t main_compute_done;
    cudaEventCreate(&main_compute_done);

    cublasOperation_t transA = CUBLAS_OP_N;
    cublasOperation_t transB = CUBLAS_OP_N;

    T falpha = 1;
    T fbeta = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float t, t1, t_Achk, t_Bchk;

    if (DEBUG) cudaEventRecord(start, stream_colchk);
    if(transA == CUBLAS_OP_N){
        encode_col_v5<T, M, K, 4><<<num_batches, dim3(M*4, 1), (M+1)*K*sizeof(T), stream_colchk>>>(num_batches,
                   dA, ldda, stridea, 
                    dA_colchk, ldda_colchk, (2*k));
    }
    else{
        // cublasSgemmStridedBatched(
        // handle, CUBLAS_OP_N, CUBLAS_OP_T, k, 2, m,
        // &falpha, dA, ldda, stridea,
        // chk_v_a, ld_chk_v, 0, &fbeta,
        // dA_rowchk, ldda_rowchk, (2*k),
        // num_batches);
        // std::cout << "  Output dA_rowchk: " << std::endl;
        
    }
    if (DEBUG) {
        cudaEventRecord(stop, stream_colchk);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&t_Achk, start, stop);
    }

    if (DEBUG) cudaEventRecord(start, stream_rowchk);
    if (transB == CUBLAS_OP_N){
        encode_row_v5<T, K, N><<<num_batches, dim3(K*2, 1, 1), 0, stream_rowchk>>>(num_batches,
                   dB, lddb, strideb, 
                    dB_rowchk, lddb_rowchk, (2*k));
    } else{
        // cublasSgemmStridedBatched(
        // handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, k, n,
        // &alpha, chk_v_b, ld_chk_v, 0,
        // dB, lddb, strideb, &fbeta,
        // dB_colchk, lddb_colchk, (2*k),
        // num_batches);
    }
    if (DEBUG) {
        cudaEventRecord(stop, stream_rowchk);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&t_Bchk, start, stop);
        t_Bchk /= 1.0;
    }

    falpha = alpha;
    fbeta = beta;

    // number of row and col of B stored in memory(no trans operation)
    int64_t mem_row = 0;
    int64_t mem_col = 0;

    if (DEBUG)  cudaEventRecord(start, stream_main);
    if (DEBUG) std::cout<<"A*B=C." << std::endl;
    if constexpr (std::is_same<T, float>::value) {
        cublasSgemmStridedBatched(
            handle, transA, transB, m, n, k,
            &falpha, dA, ldda, stridea,
            dB, lddb, strideb, &fbeta,
            dC, lddc, stridec,
            num_batches);

    } else if constexpr(std::is_same<T, half>::value) {
        cublasGemmStridedBatchedEx(
        handle, transA, transB, m, n, k,
        &falpha, dA, CUDA_R_16F, ldda, stridea,
        dB, CUDA_R_16F, lddb, strideb, &fbeta,
        dC, CUDA_R_16F, lddc, stridec,
        num_batches, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    
    cudaStreamSynchronize(stream_main);
    // std::cout << "Output dC: " << std::endl;
    // outputMatrix(dC, lddc, stridec, num_batches, m, n);
    
    if (DEBUG)  {
      cudaEventRecord(stop, stream_main);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&t, start, stop);
      printf("  gemm: %f (%f)(%f)\n", t, (double)num_batches*m*n*k*2/t/1e6, (double)num_batches*(m*k+k*n+m*n)/t/1e6);
      printf("dA_chk_gemm: %f (%f)(%f)(%f)\n", t_Achk, t_Achk/t, (double)num_batches*m*2*k*2/t_Achk/1e6, (double)num_batches*(2*k+2*m+k*m)*sizeof(T)/t_Achk/1e6);
      printf("dB_chk_gemm: %f (%f)(%f)(%f)\n", t_Bchk, t_Bchk/t, (double)num_batches*2*n*k*2/t_Bchk/1e6, (double)num_batches*(2*k+k*n+2*n)*sizeof(T)/t_Bchk/1e6);
    }

    if (DEBUG)  cudaEventRecord(start, stream_colchk);
    if(COL_FT){
      //std::cout << "  COL_FT" << std::endl;
      if (transA == CUBLAS_OP_N) {
        if (DEBUG) std::cout << "dA_colchk * dB = dC_colchk" << std::endl;;

        // K*4 must be greater then 2 * N
        update_col_v5<T, K, N, 4><<<num_batches, dim3(K*4, 1, 1), ((K+1)*N+2*K) * sizeof(T), stream_colchk>>>(num_batches,
                    dA_colchk, ldda_colchk, k*2, 
                    dB, lddb, strideb, 
                    dC_colchk, lddc_colchk, n*2);
      }
      else{
        if (DEBUG) std::cout << "dB * dA_rowchk = dC_colchk" << std::endl;
        // cublasSgemmStridedBatched(
        //     handle, transA, transB, 2, n, k,
        //     &falpha, dA_rowchk, ldda_rowchk, k*2,
        //     dB, lddb, strideb, &fbeta,
        //     dC_colchk, lddc_colchk, n*2,
        //     num_batches);
      }
      // std::cout << "Output dC_colchk: " << std::endl;
      // outputMatrixChk(dC_colchk, ldda_colchk, n*2, num_batches, 2, n);
    }
    if (DEBUG)  {
        cudaEventRecord(stop, stream_colchk);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&t1, start, stop);
        printf("  gemm-col-ft: %f (%f)(%f)(%f)\n", t1, t1/t, (double)num_batches*2*n*k*2/t1/1e6, (double)num_batches*(2*k+k*n+2*n)*sizeof(T)/t1/1e6);
    }

    if (DEBUG)  cudaEventRecord(start, stream_rowchk);
    if (ROW_FT) {
        //std::cout << "  ROW_FT" << std::endl;
        if (transB == CUBLAS_OP_N) {
          if (DEBUG) std::cout << "dA * dB_rowchk = dC_rowlchk" << std::endl;
          //we can further work on this to support trans A
            update_row_v5<T, M, K><<<num_batches, dim3(M*2, 1, 1), (2*K) * sizeof(T), stream_rowchk>>>(num_batches,
                    dA, ldda, stridea, 
                    dB_rowchk, lddb_rowchk, k*2, 
                    dC_rowchk, lddc_rowchk, m*2);
        } else{
          if (DEBUG) std::cout << "dB_colchk * dA = dC_rowlchk" << std::endl;
          // cublasSgemmStridedBatched(
          //   handle, transA, transB, m, 2, k,
          //   &falpha, dA, ldda, stridea,
          //   dB_colchk, lddb_colchk, k*2, &fbeta,
          //   dC_rowchk, lddc_rowchk, m*2,
          //   num_batches);
        }
        // std::cout << "Output dC_rowchk: " << std::endl;
        // outputMatrixChk(dC_rowchk,lddc_rowchk, m*2, num_batches, m, 2);
    }
    if (DEBUG)  {
      cudaEventRecord(stop, stream_rowchk);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&t1, start, stop);
      printf("  gemm-row-ft: %f (%f)(%f)(%f)\n", t1, t1/t, (double)num_batches*m*2*k*2/t1/1e6, (double)num_batches*(m*k+k*2+m*2)*sizeof(T)/t1/1e6);
    }

    // --- check check-sum of C---//
    if (DEBUG) std::cout << "------Check check-sum-------" << std::endl;
    if (DEBUG) cudaEventRecord(start, stream_colchk);
    if (COL_FT && CHECK_AFTER) {
      mem_row = m;
      mem_col = n;
      if (DEBUG) printf("dgemm-after-check-C-col\n");

      encode_col_v5<T, M, N, 4><<<num_batches, dim3(M*4, 1), (M+1)*N*sizeof(T), stream_colchk>>>(num_batches,
                   dC, lddc, stridec, 
                    dC_colchk_r, lddc_colchk_r, (2*n));

      T E = 1e-2;
      detect_correct_col<<<dim3(num_batches), dim3(n), 0, stream_colchk>>>(dC, lddc, E, stridec,
                                            dC_colchk,      lddc_colchk,    (2*n),
                                            dC_colchk_r,    lddc_colchk_r,  (2*n));

    }

    if (DEBUG)  {
      cudaEventRecord(stop, stream_colchk);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&t1, start, stop);
      printf("gemm-col-chk: %f (%f)(%f)(%f)\n", t1, t1/t, (double)(num_batches)*2*n*m*2/t1/1e6, (double)num_batches*(m*n+2*m+2*n)*sizeof(T)/t1/1e6);
    }

    if (DEBUG)  cudaEventRecord(start, stream_rowchk);
    if (ROW_FT && CHECK_AFTER) {
      mem_row = m;
      mem_col = n;
      if (DEBUG) printf("dgemm-after-check-C-row\n");

      encode_row_v5<T, M, N><<<num_batches, dim3(M*2, 1, 1), 0, stream_rowchk>>>(num_batches,
                   dC, lddc, stridec, 
                    dC_rowchk_r, lddc_rowchk_r, (2*m));

      T E = 1e-2;
      detect_correct_row<<<dim3(num_batches), dim3(m), 0, stream_rowchk>>>(dC, lddc, E, stridec,
                                            dC_rowchk, lddc_rowchk,     (2*m),
                                            dC_rowchk_r, lddc_rowchk_r, (2*m));

    }

    if (DEBUG)  {
        cudaEventRecord(stop, stream_rowchk);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&t1, start, stop);
        printf("gemm-row-chk: %f (%f)(%f)(%f)\n", t1, t1/t, (double)(num_batches)*m*2*n*2/t1/1e6, (double)num_batches*(m*n+2*n+2*m)*sizeof(T)/t1/1e6);
    }
}

template <typename T>
int test(){


    T *A, *B;
    T *dA, *dB, *dC;

    int64_t m = 72;
    int64_t n = 72;
    int64_t k = 64;
    int64_t num_batches = 96;

    size_t size = num_batches * m * k * sizeof(T);
    cudaMalloc((void **)&dA, size);
    A = (T *)malloc(size);
    // MaxtrixRandom(A, num_batches, m*k, m, m, k);
    cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice);
    // printf("dA: \n");
    // outputChk(dA, num_batches, m, m*k, m, k); 

    size = num_batches * k * n * sizeof(T);
    cudaMalloc((void **)&dB, size);
    B = (T *)malloc(size);
    // MaxtrixRandom(B, num_batches, k*n, k, k, n);
    cudaMemcpy(dB, B, size, cudaMemcpyHostToDevice);
    // printf("dB: \n");
    // outputChk(dB, num_batches, k, n*k, k, n);
    
    size = num_batches * m * n * sizeof(T);
    cudaMalloc((void **)&dC, size);
    cudaMemset(dC, 0, (num_batches * m * n * sizeof(T)));

    int64_t ldda_colchk = 2;
    int64_t ldda_colchk_r = 2;
    int64_t ldda_rowchk = k;
    int64_t ldda_rowchk_r = k;

    int64_t lddb_rowchk = k;
    int64_t lddb_rowchk_r = k;
    int64_t lddb_colchk = 2;
    int64_t lddb_colchk_r = 2;

    int64_t lddc_colchk = 2;
    int64_t lddc_colchk_r = 2;
    int64_t lddc_rowchk = m;
    int64_t lddc_rowchk_r = m;
    int64_t ld_chk_v = 2;

    T *dA_colchk, *dA_rowchk, *dA_colchk_r, *dA_rowchk_r;
    T *dB_colchk, *dB_rowchk, *dB_colchk_r, *dB_rowchk_r;
    T *dC_colchk, *dC_rowchk, *dC_colchk_r, *dC_rowchk_r;
    T *chk_v_a;
    T *chk_v_b;

    size = (2*num_batches) * k * sizeof(T);
    cudaMalloc((void**)&dA_colchk, size);
    cudaMemset(dA_colchk, 0, size);
    cudaMalloc((void**)&dA_colchk_r, size);
    cudaMemset(dA_colchk_r, 0, size);

    cudaMalloc((void**)&dA_rowchk, size);
    cudaMemset(dA_rowchk, 0, size);
    cudaMalloc((void**)&dA_rowchk_r, size);
    cudaMemset(dA_rowchk_r, 0, size);
    //std::cout << "  finish dA." << std::endl;
    
    cudaMalloc((void**)&dB_colchk, size);
    cudaMemset(dB_colchk, 0, size);
    cudaMalloc((void**)&dB_colchk_r, size);
    cudaMemset(dB_colchk_r, 0, size);
    
    cudaMalloc((void**)&dB_rowchk, size);
    cudaMemset(dB_rowchk, 0, size);
    cudaMalloc((void**)&dB_rowchk_r, size);
    cudaMemset(dB_rowchk_r, 0, size);
    //std::cout << "  finish dB." << std::endl;

    size = (2*num_batches) * n * sizeof(T);
    cudaMalloc((void**)&dC_colchk, size);
    cudaMemset(dC_colchk, 0, size);
    cudaMalloc((void**)&dC_colchk_r, size);
    cudaMemset(dC_colchk_r, 0, size);
    
    size = (2*num_batches) * m * sizeof(T);
    cudaMalloc((void**)&dC_rowchk, size);
    cudaMemset(dC_rowchk, 0, size);
    cudaMalloc((void**)&dC_rowchk_r, size);
    cudaMemset(dC_rowchk_r, 0, size);

    int64_t len = m;
    size = 2 * len * sizeof(T);
    cudaMalloc((void**)&chk_v_a, size);
    // std::cout << "  assign values to chk_v_a." << std::endl;
    T *h_matrix;
    h_matrix = (T *)malloc(size);
    int idx = 0;
    for(int i = 0; i < len; i++){
        idx = i*ld_chk_v;
        h_matrix[idx] = T(1);
        h_matrix[idx+1] = T(i+1);
    }
    cudaMemcpy(chk_v_a, h_matrix, size, cudaMemcpyHostToDevice);
    // std::cout << "chk_v_a: " << std::endl;
    // outputChk(chk_v_a, 1, ld_chk_v, 0, 2, m);
    free(h_matrix);

    len = n;
    size = 2 * len * sizeof(T);
    cudaMalloc((void**)&chk_v_b, size);
    // std::cout << "  assign values to chk_v_b." << std::endl;
    h_matrix = (T *)malloc(size);
    idx = 0;
    for(int i = 0; i < len; i++){
        idx = i*ld_chk_v;
        h_matrix[idx] = T(1);
        h_matrix[idx+1] = T(i+1);
    }
    cudaMemcpy(chk_v_b, h_matrix, size, cudaMemcpyHostToDevice);
    // std::cout << "chk_v_b: " << std::endl;
    // outputChk(chk_v_a, 1, ld_chk_v, 0, 2, len);
    free(h_matrix);
    //std::cout << "  finish chk_v." << std::endl;

    bool COL_FT = true;
    bool ROW_FT = true;
    bool DEBUG = false;
    bool CHECK_BEFORE = true;
    bool CHECK_AFTER = true;

    T alpha = 1;
    T beta = 0;
    int64_t stridea = m*k;
    int64_t strideb = n*k;
    int64_t stridec = m*n;
    int64_t ldda = m;
    int64_t lddb = k;
    int64_t lddc = m;

    for (int i = 0; i <10; i++) {

        auto start = high_resolution_clock::now();
        abftbgemm<T, 72, 72, 64>(m, n, k,
            alpha, dA, ldda, stridea,
            dB, lddb, strideb, beta,
            dC, lddc, stridec,
            dA_colchk, ldda_colchk,
            dA_rowchk, ldda_rowchk,
            dA_colchk_r, ldda_colchk_r,
            dA_rowchk_r, ldda_rowchk_r,
            dB_colchk, lddb_colchk,
            dB_rowchk, lddb_rowchk,
            dB_colchk_r, lddb_colchk_r,
            dB_rowchk_r, lddb_rowchk_r,
            dC_colchk, lddc_colchk,
            dC_rowchk, lddc_rowchk,
            dC_colchk_r, lddc_colchk_r,
            dC_rowchk_r, lddc_rowchk_r,
            chk_v_a, chk_v_b, ld_chk_v,
            num_batches,
            COL_FT,ROW_FT,DEBUG,CHECK_BEFORE,CHECK_AFTER);
        cudaDeviceSynchronize();
        auto stop = high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<microseconds>(stop - start);
        std::cout << "abftbgemm: " << duration.count() / 1000.0 << std::endl;
    }

    cudaFree(dA_colchk);
    cudaFree(dA_rowchk);
    cudaFree(dA_colchk_r);
    cudaFree(dA_rowchk_r);
    cudaFree(dB_colchk);
    cudaFree(dB_rowchk);
    cudaFree(dB_colchk_r);
    cudaFree(dB_rowchk_r);
    cudaFree(dC_colchk);
    cudaFree(dC_rowchk);
    cudaFree(dC_colchk_r);
    cudaFree(dC_rowchk_r);
    cudaFree(chk_v_a);
    cudaFree(chk_v_b);

    return 0;
}

int main() {
    // test<float>();
    test<half>();
}


