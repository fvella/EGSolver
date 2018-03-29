/** \file  host_csr2csc.h
 * \brief funzione di conversione da csr a csc su host
 *
 */
#ifndef FLAGHOSTCSR2CSCH

#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <errno.h>


// numrow x numcol   sparse matrix with nnz not-null int elements
void host_csr2csc(int numrow, int numcol, int nnz, int *csrValA, int *csrColIndA, int *csrRowPtrA, int *cscValA, int *cscRowIndA, int *cscColPtrA);


// all header:
#define FLAGHOSTCSR2CSCH 1
#endif

