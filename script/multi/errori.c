
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
	


//all header:
#define FLAGLOADERRORI 1
#endif
