/** \file  dev_common.h
 * \brief Header-file: definizioni lato device
 *
 */

#ifndef FLAGDEVCOMMON_H

__device__ int *dev_allData;  /**< variabile device in codice device */
__device__ int *dev_csrPtrInSuccLists;  /**< variabile device in codice device */
__device__ int *dev_csrSuccLists;  /**< variabile device in codice device */
__device__ int *dev_revSuccLists;  /**< revSuccLists[i] e' il nodo origine dell'arco che giunge a csrSuccLists[i] */
__device__ int *dev_csrPesiArchi;  /**< variabile device in codice device */
__device__ int *dev_ResNodeValues1;  /**< variabile device in codice device */
__device__ int *dev_ResNodeValues2;  /**< variabile device in codice device */
__device__ int *dev_ResNodeValuesAux;  /**< variabile device in codice device */
__device__ int *dev_flag;  /**< variabile device in codice device */
__device__ int *dev_nodeFlags1;  /**< variabile device in codice device */
__device__ int *dev_nodeFlags2;  /**< variabile device in codice device */

__device__ int *dev_transData;         /**< variabile device in codice device */
__device__ int *dev_cscPtrInPredLists; /**< variabile device in codice device */
__device__ int *dev_cscPredLists;      /**< variabile device in codice device */
__device__ int *dev_cscPesiArchiPred;  /**< variabile device in codice device */

__device__ int *dev_csrDataArchiAux; /**< variabile device in codice device */

#ifndef DEFMYLDG
	__device__ __forceinline__ int myldg(const int* ptr) {
	#if __CUDA_ARCH__ >= 350
		return __ldg(ptr);
	#else
		return *ptr;
	#endif
	}
#else
	#define DEFMYLDG 1
#endif




#define FLAGDEVCOMMON_H 1
#endif
