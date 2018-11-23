
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

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


