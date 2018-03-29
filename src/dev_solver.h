/** \file  dev_solver.h
 * \brief Header-file: funzioni device per trasferimento dati e calcolo
 *
 */

#ifndef FLAGDEVSOLVER_H


extern "C" int alloca_memoria_host();
extern "C" int alloca_memoria_device();

extern "C" int dealloca_memoria_host();
extern "C" int dealloca_memoria_device();

extern "C" int copia_dati_su_device();

/** \brief Wrapper per i vari algoritmi implementati su GPU
 *
 **/
extern "C" void gpu_solver();



#define FLAGDEVSOLVER_H
#endif
