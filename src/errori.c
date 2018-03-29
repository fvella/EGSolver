/**
 * \file errori.c
 * \brief Funzioni per la realloc di blocchi di memoria e il check dell'esito della realloc.
 *
 * \details Definite in modo grezzo dei wrapper per la realloc() e gestione dell'errore.\n
 * Le funzioni sono della forma  reallocazioneT  
 * e sono intese per reallocare un array di oggetti di tipo T, aumentandone i componenti.
 * \author Andy.
 * \todo  Codice ridondante. Migliorabile, magari usando i template.
*/


#ifndef FLAGLOADERRORI
#include "errori.h"

// *******************************************


void printWarning(const char *mess, const char *string) {
	fprintf(stderr, mess, string);fflush(stderr);
	return;
}
	
void exitWithError(const char *mess, const char *string) {
	fprintf(stderr, mess, string);fflush(stderr);
	exit(EXIT_FAILURE);
}
	

// *******************************************

void checkNullAllocation(void *ptr, const char *str) {
        if(ptr == NULL){ exitWithError("\n\nERROR in re/malloc %s\n\n", str); }
}

ulong * reallocazioneUlong(ulong newsize, ulong * old){
        ulong * r;
        r = (ulong *) realloc(old, newsize * sizeof(ulong));
	checkNullAllocation(r, "(reallocazioneUlong)");
        return r;
}

uint ** reallocazionePtrUint(ulong newsize, uint **old){
        uint **r;
        r = (uint **) realloc(old, newsize * sizeof(old));
	checkNullAllocation(r, "(reallocazionePtrUint)");
        return r;
}

int ** reallocazionePtrInt(ulong newsize, int **old){
        int **r;
        r = (int **) realloc(old, newsize * sizeof(old));
	checkNullAllocation(r, "(reallocazionePtrInt)");
        return r;
}

char ** reallocazionePtrChar(ulong newsize, char **old){
        char **r;
        r = (char **) realloc(old, newsize * sizeof(old));
	checkNullAllocation(r, "(reallocazionePtrChar)");
        return r;
}

int * reallocazioneInt(ulong newsize, int * old){
        int * r;
        r = (int *) realloc(old, newsize * sizeof(int));
	checkNullAllocation(r, "(reallocazioneInt)");
        return r;
}

uint * reallocazioneUint(ulong newsize, uint * old){
        uint * r;
        r = (uint *) realloc(old, newsize * sizeof(uint));
	checkNullAllocation(r, "(reallocazioneUint)");
        return r;
}

ushort * reallocazioneUshort(ulong newsize, ushort * old){
        ushort * r;
        r = (ushort *) realloc(old, newsize * sizeof(ushort));
	checkNullAllocation(r, "(reallocazioneUshort)");
        return r;
}



//all header:
#define FLAGLOADERRORI 1
#endif
