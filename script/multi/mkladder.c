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



#define N 6

int main(int argc, char *argv[]) {
	int val, i;
	int maxnode = 2*N-1;
        char *endptr;

	if (argc == 2) {
		errno = 0;
		val = strtol(argv[1], &endptr, 10);
		if ((errno == ERANGE && (val == LONG_MAX || val == LONG_MIN)) || (errno != 0 && val == 0)) {
			fprintf(stderr,"\nCONVERSION ERROR illegal or out-of-range parameter for option\n\n");
			exit(EXIT_FAILURE);
		}
		if ((val < 1) || (val > INT_MAX)) {
			fprintf(stderr,"\nCONVERSION ERROR illegal or out-of-range parameter for option (positive int expected)\n\n");
			exit(EXIT_FAILURE);
		}
		if (endptr == argv[1]) {
			fprintf(stderr,"\nCONVERSION ERROR illegal or missing parameter for option\n\n");
			exit(EXIT_FAILURE);
		}

		maxnode = 2*val -1;
		printf("%d;\n",maxnode);
		for (i=0; i<=maxnode; i+=2) {
			printf("%d MAX %d:%d,%d:%d;\n", i, ((i+1)%(2*val)), 0, ((i+2)%(2*val)), 0 );
		}
		for (i=0; i<=maxnode; i+=2) {
			// ORIGINALE printf("%d MIN %d:%d,%d:%d;\n", i+1, ((i+2)%(2*val)), 0, ((i+3)%(2*val)), ((i+1)==maxnode)?-1:0 );
			// penalizzando MAX
			printf("%d MIN %d:%d,%d:%d;\n", i+1, ((i+2)%(2*val)), 0, ((i+3)%(2*val)), -1 );
		}
	} else {
		fprintf(stderr,"For usage, type:  %s <number>\n(generates a ``ladder'' energy game made of 2*number nodes)\n", argv[0]);
        	fflush(stderr);
	}
}




