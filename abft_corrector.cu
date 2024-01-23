#include <cstdint>
#include <ATen/ATen.h>
#include <cuda_runtime.h>

__global__ void
colchk_detect_correct_kernel(at::Half * dA, int64_t ldda, double E, int64_t stridea,
						     at::Half * dA_colchk, 	int64_t ldda_colchk,
						     at::Half * dA_colchk_r, int64_t ldda_colchk_r){
    //determin the block to process
    dA = dA + blockIdx.x * stridea;
    dA_colchk   = dA_colchk   + blockIdx.x * 2;
    dA_colchk_r = dA_colchk_r + blockIdx.x * 2;
    
    //determine the specific colum to process
    dA = dA + threadIdx.x * ldda;
    dA_colchk   = dA_colchk   + threadIdx.x * ldda_colchk;
    dA_colchk_r = dA_colchk_r + threadIdx.x * ldda_colchk_r;
	
    double d1 = (*dA_colchk)       - (*dA_colchk_r);
    double d2 = (*(dA_colchk + 1)) - (*(dA_colchk_r + 1));
	
    //error detected
    if(fabs(d1) > E) {
    	//locate the error
		int loc = round(d2 / d1) - 1;
		printf("[col check]error detected (d1 = %f, d2 = %f, loc = %d) \n",d1, d2, loc);
			
		//the sum of the rest correct number except the error one
		double sum = 0.0;
		for(int i = 0; i < ldda; i++) {
			if (i != loc) {
				sum +=	*(dA + i); 
			}
		}
		//correct the error
		*(dA + loc) = *dA_colchk - sum;
    }
}

__global__ void
rowchk_detect_correct_kernel(at::Half * dA, int64_t ldda, double E, int64_t stridea,
						     at::Half ** dA_rowchk, 	int64_t ldda_rowchk,
						     at::Half ** dA_rowchk_r, int64_t ldda_rowchk_r){
    //determin the block to process
    dA = dA + blockIdx.x * stridea;
    dA_rowchk = dA_rowchk + blockIdx.x * 2;
    dA_rowchk_r = dA_rowchk_r + blockIdx.x * 2;
        
    //determine the specific colum to process
    dA = dA + threadIdx.x * ldda;
    dA_rowchk   = dA_rowchk   + threadIdx.x;
    dA_rowchk_r = dA_rowchk_r + threadIdx.x;
	
    double d1 = (*dA_rowchk)                 - (*dA_rowchk_r);
    double d2 = (*(dA_rowchk + ldda_rowchk)) - (*(dA_rowchk_r + ldda_rowchk_r));
	
    //error detected
    if(fabs(d1) > E) {
		//locate the error
		int loc = round(d2 / d1) - 1;
		printf("[row check]error detected (d1 = %f, d2 = %f, loc = %d) \n",d1, d2, loc);
			
		//the sum of the rest correct number except the error one
		double sum = 0.0;
		for (int i = 0; i < ldda; i++) {
		    if (i != loc) {
			sum +=	*(dA + i * ldda); 
		    }
		}
        //correct the error
		*(dA + loc * ldda) = *dA_rowchk - sum;
     }
}

void colchk_detect_correct(at::Half * dA, int64_t ldda, int64_t m, int64_t n, int64_t stridea,
				           at::Half * dA_colchk,	int64_t ldda_colchk,
				           at::Half * dA_colchk_r, 	int64_t ldda_colchk_r,
						   int64_t num_batches,
                   		   cudaStream_t stream) {
	printf("col_detect_correct called \n");
	//error threshold 
	double E = 1e-3;
	colchk_detect_correct_kernel<<<dim3(num_batches), dim3(n), 0, stream>>>(dA, ldda, E, stridea,
											dA_colchk,		ldda_colchk,
											dA_colchk_r, 	ldda_colchk_r);
}

void rowchk_detect_correct(at::Half * dA, int64_t ldda, int64_t m, int64_t n, int64_t stridea,
					 	   at::Half * dA_rowchk,		int64_t ldda_rowchk,
						   at::Half * dA_rowchk_r,		int64_t ldda_rowchk_r,
						   int64_t num_batches,
						   cudaStream_t stream) {
	printf("row_detect_correct called \n");
	//error threshold 
	
	double E = 1e-3;
	rowchk_detect_correct_kernel<<<dim3(num_batches), dim3(m), 0, stream>>>(dA, ldda, E, stridea,
											dA_rowchk, ldda_rowchk,
											dA_rowchk_r, ldda_rowchk_r);
}
