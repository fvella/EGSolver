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
	int val, i,j;
	int maxnode;
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

		maxnode = val -1;
		printf("%d;\n",maxnode);
		for (i=0; i<=maxnode; i++) {
			if (!(i%2)) {
			printf("%d MAX ", i);
			for (j=0; j<=maxnode; j++) {
				if(i!=j) { printf("%d:%d%s", j, i, ((i<maxnode)&&(j==maxnode) || ((i==maxnode)&&(j==(maxnode-1))))?";":","); }
			}
			printf("\n");
		}
		}
		for (i=0; i<=maxnode; i++) {
			if (i%2) {
			printf("%d MIN ", i);
			for (j=0; j<=maxnode; j++) {
				if(i!=j) { printf("%d:%d%s", j, -(i), ((i<maxnode)&&(j==maxnode) || ((i==maxnode)&&(j==(maxnode-1))))?";":","); }
			}
			printf("\n");
		}
		}
	} else {
		fprintf(stderr,"For usage, type:  %s <number>\n(generates a ``clique'' energy game made of nodes 0..number)\n", argv[0]);
        	fflush(stderr);
	}
}




