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
//#include <math.h>  // compilare con -lm
#include <time.h>
#include <sys/timeb.h>
#include <signal.h>
#include <libgen.h>

// numero di char max per una stringa filename
#define FILENAME_LEN 256

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




/** Struttura contenente opzioni configurazione */
typedef struct _config {

        int computationType; /**< Tipo di computazione */
        int stdinputsource; /**< flag: input from stdin (no filename) */
	char filename[FILENAME_LEN]; /**< input filename (if any) */
} config;


#define FLAGCOMMON_H 1
#endif
