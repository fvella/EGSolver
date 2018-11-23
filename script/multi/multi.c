
#include "multi.h"


/*********************************/
#include <zlib.h>  // direct access to the gzip API

// (ew!) global variable for the gzip stream.
gzFile my_gzfile = NULL;

int my_abstractread(char *buff, int buffsize) {
	// zlip usage, code from w w w . a q u a m e n t u s . c o m
	// called on a request by Flex to get the next 'buffsize' bytes of data
	// from the input, and return the number of bytes read.

	int res = gzread(my_gzfile, buff, buffsize);
	if (res == -1) {
		// file error!
		exitWithError("Error reading input (in gzread)\n", "");
	}

	return res;
}





int main(int argc, char *argv[]) {

	/* SET config param reading command-line options */	
	setconfig(argc, argv);

	parse_input();
	if (configuration.stdinputsource == 0) { gzclose(my_gzfile); }

	fflush(stdout);
	exit(EXIT_SUCCESS);
}



//*******************************************//

	
void printShortUsage(char * str) {
	fprintf(stderr,"   For usage, type:   %s  --help\n", str);
	fflush(stderr);
}



void printUsage(char * str) {
	fprintf(stderr," Usage:   %s [options]\n", str);
	fprintf(stderr," Reads an EG G and produces an EG composed by N isomorph copies of G (with renamed nodes) and,\n for each node v of the first copy, adds all edges from v to its homologues in all other copies of G.\n");
	fprintf(stderr," Options:\n");
	fprintf(stderr,"  --help  -h  -?\n\tShow this message.\n");
	fprintf(stderr,"  --copies N\n\tMake N>1 copies of the game (Default: 2)\n");
	fprintf(stderr,"  --input FILENAME\n\tReads from FILENAME instead of using stdin. (Input can be in gzipped form. Default: stdin)\n");
	fflush(stderr);
}


//***********************************************//



/* Processa opzioni: */
void setconfig(int argc, char *argv[]) {

// SET DEFAULT PARAMS
	int i;
	configuration.stdinputsource = 1;
        numCopie = 2;

	for( i = 1; i < argc; i++){
		if((strcmp(argv[i],"--help") == 0) || (strcmp(argv[i],"-?") == 0) || (strcmp(argv[i],"-h") == 0)) {
			printUsage(argv[0]);
			exit(EXIT_SUCCESS);
		}
		else if(strcmp(argv[i],"--input") == 0) {
			(configuration.filename)[0]='\0';
			checkExistsParameter(i+1, argc, "--input", argv);
			strcpy(configuration.filename,argv[++i]);
			configuration.stdinputsource = 0;
			if ((my_gzfile = gzopen(configuration.filename, "r")) == NULL) {
				fprintf(stderr,"\nERROR in opening file %s :", configuration.filename); perror(""); fprintf(stderr,"\n");
				printShortUsage(argv[0]);
				exit(1);
			}
		}
                else if(strcmp(argv[i],"--copies") == 0) {
                        i++;
                        if (checkExistsOptionalParameter(i, argc, argv) == 1) {
                                int ttemp = myatoi(argv[i], "--copies", argv);
                                if ((ttemp < 2) || (ttemp >= INT_MAX)) {
                                        fprintf(stderr,"\nERROR illegal parameter of option: --copies %s\n\n", argv[i]);
                                        printShortUsage(argv[0]);
                                        exit(1);
                                }
                                numCopie = ttemp;
                        } else {
                                i--;
                        }
                }
		else
		{
			fprintf(stderr,"\nERROR unknown option: %s\n\n", argv[i]);
			printShortUsage(argv[0]);
			exit(1);
		}
	}

// 	se manca file input apri stdin (con possibilita' che sia gzipped )
 	if (my_gzfile == NULL) {  // se NULL e allora non era presente alcuna opzione --input
// 		fprintf(stderr,"GZopening stdin...\n");fflush(stderr);
		configuration.stdinputsource = 1;
	 	if ((my_gzfile = gzdopen(fileno(stdin), "rb")) == NULL) {
 			exitWithError("Cannot gzdopen stdin\n", "");
		} //else {
 		//	fprintf(stderr,"Reading from stdin\n");fflush(stderr);
		//}
 	}
}


void checkExistsParameter (int i, int argc, char * msg, char **argv) {
	//if ((i >= argc) || (argv[i][0] == '-')) {
	if ((i >= argc) || ((argv[i][0] == '-') && (argv[i][1] == '-')) || (argv[i][0] == '|') || ((argv[i][0] == '-') && ((argv[i][1] == 'h') || (argv[i][1] == '?')))) {
		fprintf(stderr,"\nERROR missing or illegal parameter for option: %s\n\n", msg);
		printShortUsage(argv[0]);
		exit(EXIT_FAILURE);
	}
}

int checkExistsOptionalParameter (int i, int argc, char **argv) {
        if ((i >= argc) || ((argv[i][0] == '-') && (argv[i][1] == '-')) || (argv[i][0] == '|') || ((argv[i][0] == '-') && ((argv[i][1] == 'h') || (argv[i][1] == '?')))) {
                return(0);
        }
        return(1);
}


int myatoi(char *str, char *msg, char **argv) {
	char *endptr;
	long val;

	errno = 0;
	val = strtol(str, &endptr, 10);

	if ((errno == ERANGE && (val == LONG_MAX || val == LONG_MIN)) || (errno != 0 && val == 0)) {
		fprintf(stderr,"\nCONVERSION ERROR illegal or out-of-range parameter for option: %s\n\n", msg);
		printShortUsage(argv[0]);
		exit(EXIT_FAILURE);
	}

	if ((val < INT_MIN) || (val > INT_MAX)) {
		fprintf(stderr,"\nCONVERSION ERROR illegal or out-of-range parameter for option: %s  (int type expected)\n\n", msg);
		printShortUsage(argv[0]);
		exit(EXIT_FAILURE);
	}

	if (endptr == str) {
		fprintf(stderr,"\nCONVERSION ERROR illegal or missing parameter for option: %s\n\n", msg);
		printShortUsage(argv[0]);
		exit(EXIT_FAILURE);
	}

	/* If we got here, strtol() successfully parsed a number */

	//printf("strtol() returned %ld\n", val);

	if (*endptr != '\0') {       /* caratteri spuri ... */
		fprintf(stderr,"\nCONVERSION ERROR extraneous characters (%s) after parameter of option: %s\n\n", endptr,msg);
		printShortUsage(argv[0]);
		exit(EXIT_FAILURE);
	}

	return((int)val);
}


