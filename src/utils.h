/** \file utils.h
 *  \brief Header-file: Definizioni di alcune macro e strutture dati utili a diverse funzioni su device.
 */

#ifndef FLAGUTILS_H
#define FLAGUTILS_H 1

// #define THREAD_ID  ((blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x)
#define THREAD_ID (blockIdx.x * blockDim.x + threadIdx.x)



uint uplog2_int (uint val);
void checkCUDAError(const char *msg);

/** \brief Accede all'HW e determina quanti device sono disponibili. In caso accede a alcune properties del device 0
 *  \param deviceCount device disponibili (attualmente viene usato solo il device con ID=0)
 *  \param conf struct con le opzioni derivanti dalle command-line options
 */
extern "C" void checkDevice(int *deviceCount, config *conf);

#endif
