// nvcc csr2csc.cu  -o csr2csc  -lcusparse -L /usr/local/cuda/lib64 -lcuda -lcudart -lm  -lstdc++  -I /usr/local/cuda/include
//-------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cusparse.h"

cusparseStatus_t status;
cusparseHandle_t handle;
cusparseMatDescr_t descr;

void init_library() {
    /* initialize cusparse library */
	status= cusparseCreate(&handle);
	if (status != CUSPARSE_STATUS_SUCCESS) {
		printf ("init_converter: cusparseCreate failed \n"); 
		cudaDeviceReset();
		fflush (stdout);
		exit(2);
	}

}
void init_converter() {
	/* create and setup matrix descriptor */ 
	status= cusparseCreateMatDescr(&descr); 
	if (status != CUSPARSE_STATUS_SUCCESS) {
		printf ("init_converter: cusparseCreateMatDescr failed \n");
		cudaDeviceReset();
		fflush (stdout);
		exit(2);
	}       
	cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);  
}


void exit_converter() {
    cusparseDestroyMatDescr(descr);
}


void exit_library() {
    cusparseDestroy(handle);
}



/*
Da nvidia forum: 
"The CUSPARSE library function csr2csc allocates an extra array of size nnz*sizeof(int) to store temporary data.
The nnz stands for the number of non-zero elements and should match the index stored in csrRowPtr[last_row+1] as usual in CSR format."
*/

void csr2csc(int numrow, int numcol, int nnz, int* dev_csrValA, int* dev_csrRowPtrA, int* dev_csrColIndA, int* dev_cscValA, int* dev_cscRowIndA, int* dev_cscColPtrA) {

    /* conversion routines (convert matrix from CSR 2 CSC format) */
	status = cusparseScsr2csc(handle, numrow, numcol, nnz, (float*)dev_csrValA, dev_csrRowPtrA, dev_csrColIndA, (float*)dev_cscValA, dev_cscRowIndA, dev_cscColPtrA, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO);
	if (status != CUSPARSE_STATUS_SUCCESS) {
		////cudaError_t err = cudaGetLastError();
		printf ("csr2csc: cusparseScsr2csc failed \n"); 
		////fprintf(stderr, "ERRORE CUDA in csr2csc:  >%s<.  Eseguo:EXIT\n",  cudaGetErrorString(err) );
		cudaDeviceReset();
		fflush (stdout);
		exit(2);
	}
}




/* TEST CODE 
int main(){     
    cudaError_t cudaStat1,cudaStat2,cudaStat3;

    int * host_csrValA=0;
    int * host_csrRowPtrA=0;
    int * host_csrColIndA=0;

    int * host_cscValA=0;
    int * host_cscRowIndA=0;
    int * host_cscColPtrA=0;

    int * dev_csrValA=0;
    int * dev_csrRowPtrA=0;
    int * dev_csrColIndA=0;

    int * dev_cscValA=0;
    int * dev_cscRowIndA=0;
    int * dev_cscColPtrA=0;

    int nnz=9;
    int numcol=5;
    int numrow=4;

    if ((host_csrValA = (int *)malloc(nnz*sizeof(host_csrValA[0])) )== NULL ) {printf("errore 1:\n"); fflush(stdout); exit(1);}; 
    if ((host_csrRowPtrA = (int *)malloc((numrow+1)*sizeof(host_csrRowPtrA[0])) )== NULL ) {printf("errore 1:\n"); fflush(stdout); exit(1);}; 
    if ((host_csrColIndA = (int *)malloc(nnz*sizeof(host_csrColIndA[0])) )== NULL ) {printf("errore 1:\n"); fflush(stdout); exit(1);}; 

    host_csrValA[0] = 1;
    host_csrValA[1] = 4;
    host_csrValA[2] = 2;
    host_csrValA[3] = 3;
    host_csrValA[4] = 5;
    host_csrValA[5] = 7;
    host_csrValA[6] = 8;
    host_csrValA[7] = 9;
    host_csrValA[8] = 6;

    host_csrRowPtrA[0]= 0;
    host_csrRowPtrA[1]= 2;
    host_csrRowPtrA[2]= 4;
    host_csrRowPtrA[3]= 7;
    host_csrRowPtrA[4]= 9;

    host_csrColIndA[0] = 0;
    host_csrColIndA[1] = 1;
    host_csrColIndA[2] = 1;
    host_csrColIndA[3] = 2;
    host_csrColIndA[4] = 0;
    host_csrColIndA[5] = 3;
    host_csrColIndA[6] = 4;
    host_csrColIndA[7] = 2;
    host_csrColIndA[8] = 4;


    host_cscValA = (int *)malloc(nnz*sizeof(host_cscValA[0])); 
    host_cscRowIndA = (int *)malloc(nnz*sizeof(host_cscRowIndA[0])); 
    host_cscColPtrA = (int *)malloc((numcol+1)*sizeof(host_cscColPtrA[0])); 

    printf("testing example\n");
    //print the matrix
    printf("Input data:\n");
    int i;
    for (i=0; i<nnz; i++){        
        printf("csrValA[%d]=%d   ",i,(int)host_csrValA[i]);
    }  printf("\n");fflush(stdout);
    for (i=0; i<(numrow+1); i++){        
        printf("csrRowPtrA[%d]=%d  ",i,host_csrRowPtrA[i]);
    }  printf("\n");fflush(stdout);
    for (i=0; i<nnz; i++){        
        printf("csrColIndA[%d]=%d  ",i,host_csrColIndA[i]);
    }  printf("\n");fflush(stdout);


    // allocate GPU memory and copy the matrix and vectors into it 
    cudaStat1 = cudaMalloc((void**)&dev_csrValA,nnz*sizeof(dev_csrValA[0])); 
    cudaStat2 = cudaMalloc((void**)&dev_csrRowPtrA,(numrow+1)*sizeof(dev_csrRowPtrA[0]));
    cudaStat3 = cudaMalloc((void**)&dev_csrColIndA,     nnz*sizeof(dev_csrColIndA[0])); 
    if ((cudaStat1 != cudaSuccess) || (cudaStat2 != cudaSuccess) || (cudaStat3 != cudaSuccess)) {
        printf("Device malloc failed csr");
        return 1; 
    }    
    cudaStat1 = cudaMalloc((void**)&dev_cscValA,nnz*sizeof(dev_cscValA[0])); 
    cudaStat2 = cudaMalloc((void**)&dev_cscRowIndA,nnz*sizeof(dev_cscRowIndA[0]));
    cudaStat3 = cudaMalloc((void**)&dev_cscColPtrA,     (numcol+1)*sizeof(dev_cscColPtrA[0])); 
    if ((cudaStat1 != cudaSuccess) || (cudaStat2 != cudaSuccess) || (cudaStat3 != cudaSuccess)) {
        printf("Device malloc failed csc");
        return 1; 
    }    

    cudaStat1 = cudaMemcpy(dev_csrValA, host_csrValA, (size_t)(nnz*sizeof(host_csrValA[0])), cudaMemcpyHostToDevice);
    cudaStat2 = cudaMemcpy(dev_csrRowPtrA, host_csrRowPtrA, (size_t)((numrow+1)*sizeof(host_csrRowPtrA[0])), cudaMemcpyHostToDevice);
    cudaStat3 = cudaMemcpy(dev_csrColIndA,      host_csrColIndA,      (size_t)(nnz*sizeof(host_csrColIndA[0])),      cudaMemcpyHostToDevice);
    if ((cudaStat1 != cudaSuccess) || (cudaStat2 != cudaSuccess) || (cudaStat3 != cudaSuccess)) {
        printf("Memcpy from Host to Device failed");
        return 1;
    }
    
    init_library();
    init_converter();
    
    csr2csc(numrow, numcol, nnz, dev_csrValA, dev_csrRowPtrA, dev_csrColIndA, dev_cscValA, dev_cscRowIndA, dev_cscColPtrA);



    cudaStat1 = cudaMemcpy(host_cscValA, dev_cscValA, (size_t)(nnz*sizeof(dev_cscValA[0])), cudaMemcpyDeviceToHost);
    cudaStat2 = cudaMemcpy(host_cscRowIndA, dev_cscRowIndA, (size_t)(nnz*sizeof(dev_cscRowIndA[0])), cudaMemcpyDeviceToHost);
    cudaStat3 = cudaMemcpy(host_cscColPtrA,      dev_cscColPtrA,      (size_t)((numcol+1)*sizeof(dev_cscColPtrA[0])),      cudaMemcpyDeviceToHost);
    if ((cudaStat1 != cudaSuccess) || (cudaStat2 != cudaSuccess) || (cudaStat3 != cudaSuccess)) {
        printf("Memcpy from Device to Host failed");
        return 1;
    }
    
    printf("Output data:\n");
    for (i=0; i<nnz; i++){        
        printf("cscValA[%d]=%d   ",i,(int)host_cscValA[i]);
    }  printf("\n");fflush(stdout);
    for (i=0; i<nnz; i++){        
        printf("cscRowIndA[%d]=%d  ",i,host_cscRowIndA[i]);
    }  printf("\n");fflush(stdout);
    for (i=0; i<(numcol+1); i++){        
        printf("cscColPtrA[%d]=%d  ",i,host_cscColPtrA[i]);
    }  printf("\n");fflush(stdout);

    exit_converter();
    exit_library();
}



*/
