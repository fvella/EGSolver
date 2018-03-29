/** \file  host_csr2csc.c
 * \brief funzione di conversione da csr a csc su host
 *
 */
#ifndef FLAGHOSTCSR2CSC



#include "host_csr2csc.h"




// numrow x numcol   sparse matrix with nnz not-null int elements
void host_csr2csc(int numrow, int numcol, int nnz, int *csrValA, int *csrColIndA, int *csrRowPtrA, int *cscValA, int *cscRowIndA, int *cscColPtrA) {
	int i, j, k, l;
	int *ptr;

	for (i=0; i<=numcol; i++) {
		cscColPtrA[i] = 0;
	}

	/* determine column lengths */
	for (i=0; i<nnz; i++) {
		cscColPtrA[csrColIndA[i]+1]++;
	}

	for (i=0; i<numcol; i++) {
		cscColPtrA[i+1] += cscColPtrA[i];
	}

	/* write csc */
	for (i=0, ptr=csrRowPtrA; i<numrow; i++, ptr++) {
		for (j=*ptr; j<*(ptr+1); j++){
			k = csrColIndA[j];
			l = cscColPtrA[k]++;
			cscRowIndA[l] = i;
			cscValA[l] = csrValA[j];
		}
	}
	/* shift back cscColPtrA */
	for (i=numcol; i>0; i--) {
		cscColPtrA[i] = cscColPtrA[i-1];
	}
	cscColPtrA[0] = 0;
}


/* TEST CODE 

int main(){     

    int * host_csrValA=0;
    int * host_csrRowPtrA=0;
    int * host_csrColIndA=0;

    int * host_cscValA=0;
    int * host_cscRowIndA=0;
    int * host_cscColPtrA=0;

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


     host_csr2csc(numrow, numcol, nnz, host_csrValA, host_csrColIndA, host_csrRowPtrA, host_cscValA, host_cscRowIndA, host_cscColPtrA);


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

}



*/

// all header:
#define FLAGHOSTCSR2CSC 1
#endif


