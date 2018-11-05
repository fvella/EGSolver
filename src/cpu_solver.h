/** \file  cpu_solver.h
 * \brief Header-file per calcolo su CPU
 *
 */

#ifndef FLAGCPU_SOLVER_H

#include "common.h"
#include "errori.h"
#include "host_csr2csc.h"

#include <assert.h>

/** \brief Wrapper per i vari algoritmi implementati su CPU
 *
 **/
void cpu_solver();

long aggiorna_max_loop(long Narchi, long Nnodi, long MGpesi, long maxloop);

#define FLAGCPU_SOLVER_H 1
#endif
