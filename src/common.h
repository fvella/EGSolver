/** \file  common.h
 * \brief Header-file: Varie funzioni di output (nomi esplicativi)
 *
 */

#ifndef FLAGCOMMON_H

#undef ASSERT_FLAG
#undef MYDEBUG
#undef CHECKCUDAERR_MEM
#undef CHECKCUDAERR_KER
#undef ALL_RUNTIME_CHECKS
#undef CHECKCUDASAFE




#define BIT32ONARCH64BIT
//#define BIT64ONARCH64BIT

#ifdef BIT64ONARCH64BIT
/** Wrapper per la funzione "ffsll(): find first bit set in a word" */
        #define MY_DEVFFSLL(X) __ffsll((X))
/** Wrapper per la funzione device "popcll(): find number of set bits in a word" */
        #define MY_DEVPOPLL(X) __popcll((X))
/** Wrapper per la funzione host "popcll(): find number of set bits in a word" */
        #define MY_HSTPOPLL(X) __builtin_popcountl((X))
        #define MY_HSTCLZ(X) __builtin_clzl((X))
        #define MY_DEVCLZ(X) __clzl((X))
        #define A_BIG_NUM  ULONG_MAX
        #define MY_HSTNEXT2POW(X) ((1)<<((__builtin_popcountl(X)>1?1:0) +(SIZEULONG-1) - __builtin_clzl(X)))
        #define MY_DEVNEXT2POW(X) ((1)<<((__popcll(X)>1?1:0) +(SIZEULONG-1) - __clzll(X)))
#else
/** Wrapper per la funzione "ffs(): find first bit set in a word" */
        #define MY_DEVFFSLL(X) __ffs((X))
/** Wrapper per la funzione device "popc(): find number of set bits in a word" */
        #define MY_DEVPOPLL(X) __popc((X))
/** Wrapper per la funzione host "popc(): find number of set bits in a word" */
        #define MY_HSTPOPLL(X) __builtin_popcount((X))
        #define MY_HSTCLZ(X) __builtin_clz((X))
        #define MY_DEVCLZ(X) __clz((X))
        #define A_BIG_NUM  UINT_MAX
        #define MY_HSTNEXT2POW(X) ((1)<<((__builtin_popcount(X)>1?1:0) +(SIZEULONG-1) - __builtin_clz(X)))
        #define MY_DEVNEXT2POW(X) ((1)<<((__popc(X)>1?1:0) +(SIZEULONG-1) - __clz(X)))
#endif


/* ************ CONTROLLI PER DEVELOPEMENT & DEBUG **************** */

// attiva gli assert()
//#define ASSERT_FLAG 1

// attiva alcuni controlli al runtime
//#define MYDEBUG 1

// attiva tutti controlli al runtime
//#define ALL_RUNTIME_CHECKS 1


// se definito aggiunge piu' controlli checkCUDAError() synchronize dopo kernel e memcopy
// OSS: per ora controlla quasi tutti i cudaThreadSynchronize() e cudaDeviceSynchronize() 
// quelli dopo i kernel (_KER) servono per il catch dell'errore
// alcuni di quelli dopo i memcpy (_MEM) potrebbeo esser necessari
//#define CHECKCUDAERR_MEM 1
//#define CHECKCUDAERR_KER 1

// se definito aggiunge il controllo sull'esito di molte delle chiamate a CUDA relative a allocazione memoria
//#define CHECKCUDASAFE 1


#ifdef ALL_RUNTIME_CHECKS
        #define ASSERT_FLAG 1
        #define CHECKCUDAERR_MEM 1
        #define CHECKCUDAERR_KER 1
        #define CHECKCUDASAFE 1
#endif


#ifdef CHECKCUDASAFE
        #define CHECKCUDAERR 1
        #define CUDASAFE(A, B)   if( (A) != cudaSuccess) { fprintf(stderr,"ERROR: %s : %s\n", (B), cudaGetErrorString( (A) ));  exit(-1);  }
#endif
#ifndef CHECKCUDASAFE
        #define CUDASAFE(A, B)  A
#endif

/**< Macro che, se attivato CHECKCUDASAFE, esegue un check di eventuali errori occorsi durante chiamate al device. */
#ifdef CHECKCUDAERR_MEM
        #define DEVSYNCANDCHECK_MEM(A)   {cudaDeviceSynchronize(); checkCUDAError(A);}
#endif
#ifndef CHECKCUDAERR_MEM
        #define DEVSYNCANDCHECK_MEM(A)  ;
#endif
#ifdef CHECKCUDAERR_KER
        #define DEVSYNCANDCHECK_KER(A)   {cudaDeviceSynchronize(); checkCUDAError(A);}
#endif
#ifndef CHECKCUDAERR_KER
        #define DEVSYNCANDCHECK_KER(A)  ;
#endif


/* ******************************************************* */




#ifndef _GNU_SOURCE
        #define _GNU_SOURCE 1
#endif

#include <limits.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <sys/wait.h>
#include <math.h>  // compilare con -lm
#include <time.h>
#include <sys/timeb.h>
#include <signal.h>
#include <libgen.h>


#if defined(__APPLE__)
        #include <vector_types.h>
        #include <cuda.h>
        // In alcuni sistemi serve anche   #include <cuda_runtime.h>
#else
        #include <cuda.h>
        //#include <cuda_runtime_api.h>
        #include <cuda_runtime.h>
#endif


/* costanti e parametri del solver */
#define GAMETYPE_PARITY 0
#define GAMETYPE_MPG 1
#define GAMETYPE_MPG_ARENA 2

/** Device id del device usato */
#define DEFAULT_DEVICE 0

#define CPU_COMPUTATION 0
#define GPU_COMPUTATION 1
#define DEFAULT_COMPUTATION_UNIT CPU_COMPUTATION

#define DEFAULT_THREADSPERBLOCK 256
//#define DEFAULT_THREADSPERBLOCK 512

// numero di blocchi per kernel (si usa una grid con solo una dimensione)
#define MAX_BLOCKPERKERNEL 32768

// numero di char max per una stringa filename
#define FILENAME_LEN 256

// algoritmo da usare per mpg
#define ALGOR_EG 6
#define ALGOR_EG0 7
#define COMPARA 8
#define DEFAULT_ALGOR ALGOR_EG

// 2^DEFAULT_LOOP_SLICE e' il numero di loop per ZP in un lancio di un kernel.
// Se servono piu' loop si rilancia piu' volte il kernel
#define DEFAULT_LOOP_SLICE 17
#define DEFAULT_LOOP_SLICE_FOR_EG 0

// soluzione o cleaning dell'input
#define TASK_TRANS_INPUT 0
#define TASK_SOLVE_MPG 1
#define DEFAULT_TASK TASK_TRANS_INPUT

// output solution in a single text line (default no)
#define NO_OUTPUT 0
#define YES_ONELINEOUT 1
#define NO_ONELINEOUT 2

// modo di aggiungere weight se l'input descrive un parity game
#define NOT_ADD_WEIGHTS 0
#define ADD_UNIT_WEIGHTS 1
#define ADD_EXP_WEIGHTS 2
#define ADD_RND_WEIGHTS 3
#define DEFAULT_ADD_WEIGHTS NOT_ADD_WEIGHTS
#define DEFAULT_ADD_RND_WEIGHTS_VAL 1
#define MAX_RAND_NUM  (RAND_MAX/2)
#define NO_USER_RND_SEED  (-1)

/** Nessun timeout */
#define UNSET_TIMEOUT_OPT 0
/** Timeout attivo */
#define SET_TIMEOUT_OPT 1
/** Valore di default per timeout (sec) */
#define DEFAULT_TIMEOUT_SEC 60

/** Valore di default per max_loop_opt_val. Il numero di loop per --zp, --eg, ... */
#define DEFAULT_MAX_LOOP_VAL (-1)

/** lunghezza dell'istogramma dei degree */
#define LEN_DEGREEHIST 128
#define NO_PRINT_DEGREEHIST 0
#define YES_PRINT_DEGREEHIST 1
#define DEFAULT_PRINT_DEGREEHIST NO_PRINT_DEGREEHIST


/** criteri di sorting dei nodi */
#define SORT_N 0
#define SORT_I 1
#define SORT_O 2
#define SORT_A 3
#define SORT_OI 4
#define DEFAULT_NODESORT SORT_N

//MACRO:
#ifndef MAX
        #define MAX( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef MIN
        #define MIN( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#ifndef SIGN
	#define SIGN(A) ((A) > 0 ? +1 : ((A) < 0 ? -1 : 0))
#endif

//arrotondamento all'intero superiore
#ifndef MYCEIL
        #define MYCEIL( a, b ) ( (((a) / (b)) + ( (((a) % (b)) == 0) ? 0 : 1 )) )
#endif
// il minimo multiplo di b maggiore o uguale ad a:
#ifndef MYCEILSTEP
        #define MYCEILSTEP( a, b ) ( (((a) / (b)) + ( (((a) % (b)) == 0) ? 0 : 1 )) * b )
#endif

/** Test per verificare se X sia una potenza di 2 */
#ifndef IS2POW
	#define IS2POW(X) ( ((X) != 0) && (!((X) & ((X)-1))) )
#endif

/** operatore usato nell'algoritmo EG
 * \todo Da ottimizzare 
 * \warning Riferisce la variabile globale MG_pesi
 */
#ifndef OMINUS
	#define OMINUS(A,B) ( (((A)<INT_MAX) && (((A)-(B))<=(MG_pesi))) ? ((0<((A)-(B))) ? ((A)-(B)) : (0)) : (INT_MAX) )
#endif




typedef unsigned char uchar;
typedef unsigned int uint;
typedef unsigned long ulong;
typedef unsigned short ushort;





/* definizioni per flex/bison */
extern FILE *yyin;
extern int linenum;


//PROTOTIPI
void header_proc(int max);
void footer_proc();
void body_line_proc1(int nomeEsterno, int priority, int owner);
void body_line_proc2();
void thename_proc(char *name);
void successor_proc(int succ);
void successor_procP(int succ);

void mpg_header_proc(int max);
void mpg_footer_proc();
void mpg_body_line_proc1(int nomeEsterno, int priority, int owner);
void mpg_body_line_proc2();
void mpg_thename_proc(char *name);
void mpg_successor_proc(int succ);
void mpg_successor_procP(int succ);
void mpg_weight_proc(int w);
void mpg_weight_procP(int w);




/** Struttura che raccoglie statistiche sull'esecuzione  */
typedef struct _stat {
        double inputtime; /**< Tempo impiegato per leggere l'input */
        double alloctime; /**< Tempo impiegato per allocare spazio e trasferire dati su device */
        double solvingtime; /**< Tempo impiegato per risolvere istanza */
        uint numkernelscalls;  /**< Numero di invocazioni di kernel */
        ulong device_usedGlobalMem; /**< Quantita' di memoria global allocata sul device */
} stat;


/** Struttura contenente opzioni configurazione */
typedef struct _config {

        int computationType; /**< Tipo di computazione */
        int algoritmo; /**< Algoritmo per calcolo */
        int stdinputsource; /**< flag: input from stdin (no filename) */
	char filename[FILENAME_LEN]; /**< input filename (if any) */

        int onelineout; /**< stampa la soluzione su una singola linea (a volte utile per text-post-processing) */
        int add_weights_mode; /**< modo di aggiungere weights a pg da usare come mpg */
	int rndWeightLow; /**< intervallo per i pesi generati casualmente */
	int rndWeightHigh; /**< intervallo per i pesi generati casualmente */
	int rndSeed; /**< seed per pseudo-random generator */

        int deviceid; /**< Previsto per sistemi con piu' devices; attualmente si usa il device num DEFAULT_DEVICE */
        char devicename[256]; /**< Il device usato */
        ushort warpSize;  /**< Dimensione del wrap */
        int capabilityMajor;  /**< compute capability major revision number del device usato */
        int capabilityMinor;  /**< compute capability minor revision number del device usato */
        size_t deviceProp_totalGlobalMem;  /**< Quantita' di ram disponibile sul device; Numero di byte rilevati da cudaGetDeviceProperties() */
        size_t sharedMemPerBlock; /**< Quantita' di shared memory per block */
        uint maxThreadsPerBlock; /**< Numero massimo di threads per block */
        int clockRate; /**< Clock della GPU del device usato*/
        int ECCEnabled; /**< Flag su ECC support del device usato */

        uint threadsPerBlock; /**< Numero di thread per block */
        uint deviceCount; /**< Numero di device rilevati  */

        long max_loop_val;
        int loop_slice;
        int loop_slice_for_EG;

        uint nodesorting;

        uint timeoutOpt;
        uint timeoutSeconds;

        int print_degree_hist;
} config;


#define FLAGCOMMON_H 1
#endif
