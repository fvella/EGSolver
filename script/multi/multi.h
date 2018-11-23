#ifndef FLAGMULTI_H

#include "common.h"
#include "errori.h"
#include "parser.h"

#include <assert.h>


char str[256];


// CSR zero-based (i nodi sono interi e 0 e' un possibile nodo)
int max_nodo;             /**< massimo numero/nodo (i nodi sono gli interi 0..num_nodi) */
int multi_max_nodo;
config  configuration;

int numSucc;
int lastNode;
int lastOwner;
int numCopie;
int successori[1048576];
int pesi[1048576];

void parse_input();

void setconfig(int argc, char *argv[]);

void checkExistsParameter (int i, int argc, char * msg, char **argv);

int checkExistsOptionalParameter (int i, int argc, char **argv);

int myatoi(char *str, char *msg, char **argv);



#define FLAGMULTI_H 1
#endif
