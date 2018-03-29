/** \file utils.cu
 *  \brief Definizioni di alcune macro e strutture dati utili a diverse funzioni su device.
 */

#ifndef FLAGUTILS_CU
#define FLAGUTILS_CU 1


#include <cuda.h>
#include <cuda_runtime.h>

#include "common.h"
#include "utils.h"




/** @cond NONDOC */ // non doxycumentati
extern "C" void checkNullAllocation (void *ptr, const char *str);
extern "C" int **reallocazionePtrInt(ulong newsize, int ** old);
extern "C" void exitWithError(const char *mess, const char *string);
/** @endcond */



uint uplog2_int (uint val) {
        uint res = 0;
        if (val <= 0) return(0);
        while (val != 0) {
                val >>= 1;
                res++;
        }
        return(res);
}



void checkCUDAError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        fprintf(stderr, "ERRORE CUDA: >%s<: >%s<.  Eseguo:EXIT\n", msg, cudaGetErrorString(err) );
        exit(-1);
    }                         
}





void checkDevice(int *deviceCount, config *conf) {
	cudaError_t error_id = cudaGetDeviceCount(deviceCount);
	if (error_id != cudaSuccess) {
		exitWithError("cudaGetDeviceCount returned: >%s<\n", cudaGetErrorString(error_id));
	}
	if ((*deviceCount) > (conf->deviceid)) { // il device scelto
//		fprintf(stderr,"\nOOOOOOOOOOOOO  deviceCount:%d  deviceid:%d)\n\n", (*deviceCount), conf->deviceid);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, conf->deviceid);
		conf->deviceProp_totalGlobalMem = deviceProp.totalGlobalMem;
		conf->maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
		conf->sharedMemPerBlock = deviceProp.sharedMemPerBlock;
		conf->warpSize = deviceProp.warpSize;
		conf->capabilityMajor = deviceProp.major;
		conf->capabilityMinor = deviceProp.minor;
		conf->clockRate = deviceProp.clockRate;
		conf->ECCEnabled = deviceProp.ECCEnabled;
		//conf->multiProcessorCount = deviceProp.multiProcessorCount;
		memcpy(conf->devicename, deviceProp.name, 256*sizeof(char));
	} else {
		exitWithError("checkDevice returned: >%s<\n", "Selected device not available");
	}
}





#endif
