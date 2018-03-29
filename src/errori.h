/**
 * \file errori.h
 * \brief Header-file: Funzioni per la realloc di blocchi di memoria e il check dell'esito della realloc.
 *
 * \details Definite in modo grezzo dei wrapper per la realloc() e gestione dell'errore.\n
 * Le funzioni sono della forma  reallocazioneT  
 * e sono intese per reallocare un array di oggetti di tipo T, aumentandone i componenti.
 * \author Andy.
 * \todo  Codice ridondante. Migliorabile, magari usando i template.
*/

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>


/** \brief Reallocazione per un array di #ulong
 *
 * \param newsize Nuova dimensione dell'array
 * \param old Puntatore al vecchio array
 * \return Puntatore al nuovo array
 */
ulong *reallocazioneUlong(ulong newsize, ulong * old);

uint ** reallocazionePtrUint(ulong newsize, uint ** old);
int ** reallocazionePtrInt(ulong newsize, int ** old);
char ** reallocazionePtrChar(ulong newsize, char ** old);
int *reallocazioneInt(ulong newsize, int * old);
uint *reallocazioneUint(ulong newsize, uint * old);
ushort *reallocazioneUshort(ulong newsize, ushort * old);

/** \brief Controlla se il puntatore e' NULL, in tal caso invoca exitWithError()
 *
 * \param ptr Puntatore a memoria allocata
 * \param str stringa passata a exitWithError() in caso ptr==NULL
 */
void checkNullAllocation (void *ptr, const char *str);


/** \brief Stampa un messaggio 
 *
 * \param mess prima parte del messaggio
 * \param string seconda parte del messaggio
 */
void printWarning(const char *mess, const char *string);


/** \brief Stampa un messaggio ed invoca exit(EXIT_FAILURE)
 *
 * \param mess prima parte del diagnostico
 * \param string seconda parte del diagnostico
 */
void exitWithError(const char *mess, const char *string);


