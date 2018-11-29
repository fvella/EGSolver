/*! \mainpage egsolver Documentation
 * 
 * \section intro_sec User Manual
 *
 * ... to be done....
 *
 * etc...
 *
 *
 */




/** \file egsolver.c
 * \brief  File principale del solver.
 *
 * \todo Definire le strategie di aggiunta dei weights quando l'input e' un PG
 *
 * \todo Verifica del risultato
 *
 * \todo Verificare se serve gestire ulteriori statistiche
 *
 * \todo Sostituire uso deprecated di ftime() con altra soluzione
 *
 */


#include "egsolver.h"
#include "cpu_solver.h"
//#include "dev_common.h"
#include "sorting_criteria.h"

/** @cond NONDOC */ // non doxycumentati
//PROTOTIPI da altri sorgenti:
extern void checkDevice(int *deviceCount, config* conf);
extern void checkCUDAError(const char *msg);
/** @endcond */


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

/*********************************/

/* Calcola il bound al numero dei loop. Se maggiore di LONG_MAX, warning e usa LONG_MAX-1 */
long aggiorna_max_loop(long Narchi, long Nnodi, long MGpesi, long maxloop) {

        char str[256];
	long res = (maxloop);


        if ((res) < ((long)0)) { // default (-1) indica un numero max pari al teorico 
                //CHECK OVERFLOW IN:  res = Narchi*MGpesi+1;
        //      if (MGpesi > (LONG_MAX-1)/Narchi) {
        //              // overflow handling
        //              sprintf(str,"%d * %d", Narchi, MGpesi);
        //              exitWithError("Error too many loops: %s --> overflow\n", str);
        //      } else {
        //              res = Narchi*MGpesi+1; 
                if (MGpesi > (LONG_MAX-1)/(Nnodi*Narchi)) {  //TODO: sovrastimo il numero dei loop
                        // overflow handling
                        //sprintf(str,"%ld * %ld * %ld > %ld", Nnodi, Narchi, MGpesi, LONG_MAX);
                        //exitWithError("Error too many loops: %s --> overflow\n", str);
                        res = LONG_MAX-1;
                        sprintf(str,"%ld * %ld * %ld > %ld --> using LONG_MAX-1:%ld", Nnodi, Narchi, MGpesi, LONG_MAX, res);
                        printWarning("WARNING too many loops: %s\n", str);
                } else {
                        res = Nnodi * Narchi * MGpesi + 1;
                }
        }
        return(res);
}

/*********************************/

void output_solution_singleline() {
	int idx;
	for (idx=0; idx<num_nodi; idx++) { printf("V(%d)=%d\t", nomeExt_of_nomeInt[mapping[idx]], host_ResNodeValues1[idx]); }
	printf("\n");
}

void output_solution_onenodeperline() {
	int idx;
	// RIDONDA: for (idx=0; idx<num_nodi; idx++) { printf("%d : %d(%d--%d)[%d](%d)\t%d\n", nomeExt_of_nomeInt[mapping[idx]], idx, nomeExt_of_nomeInt[idx], nomeInt_of_nomeExt[idx], mapping[nomeExt_of_nomeInt[idx]], mapping[nomeInt_of_nomeExt[idx]], host_ResNodeValues1[idx]); }
	for (idx=0; idx<num_nodi; idx++) { printf("%d)\t%d\n", nomeExt_of_nomeInt[mapping[idx]], host_ResNodeValues1[idx]); }
}


void remap_instance() {
	int idy;
	int nomeInt, nodoDest, arco;

	if ((input_gametype == GAMETYPE_PARITY) || (input_gametype == GAMETYPE_MPG) || (input_gametype == GAMETYPE_MPG_ARENA)) {
		csrPtrInSuccLists[counter_nodi] = num_archi;

		for (nomeInt=0; nomeInt < counter_nodi; nomeInt++) { // calcolo out-degree
			outDegrees_of_csr[nomeInt] = csrPtrInSuccLists[nomeInt+1] - csrPtrInSuccLists[nomeInt];
		}
		for (arco=0; arco < num_archi; arco++) { // calcolo in-degree
			nodoDest = csrSuccLists[arco];
			inDegrees_of_csr[nomeInt_of_nomeExt[nodoDest]]++;
		}

//		printf("Degrees:\nNodo\t Out\t In:\n");
//		for (nomeInt=0; nomeInt < counter_nodi; nomeInt++) { printf("%d \t %d \t %d\n", nomeInt, outDegrees_of_csr[nomeInt], inDegrees_of_csr[nomeInt]); }
//
//		printf("\nindice   :");
//		for (nomeInt=0; nomeInt < counter_nodi; nomeInt++) { printf("%d \t",nomeInt); }
//		printf("\nIntofExt :");
//		for (nomeInt=0; nomeInt < counter_nodi; nomeInt++) { printf("%d \t",nomeInt_of_nomeExt[nomeInt]); }
//		printf("\nExtofInt :");
//		for (nomeInt=0; nomeInt < counter_nodi; nomeInt++) { printf("%d \t",nomeExt_of_nomeInt[nomeInt]); }
//		printf("\n");



// Riordina i nodi mettendo prima quelli dell'owner 0:
// mapping[nomeInternoSorted] = nomeOrdineLettura  dove nomeInterno mette prima gli owner=0
// revmapping[nomeOrdineLettura] = nomeInternoSorted
		mapping = (int *)malloc((1+num_nodi)*sizeof(int)); checkNullAllocation(mapping,"allocazione mapping");
		revmapping = (int *)malloc((1+num_nodi)*sizeof(int)); checkNullAllocation(revmapping,"allocazione revmapping");

		for (idy=0; idy < counter_nodi; idy++) {
			mapping[idy] = idy;
		}

/** criteri di sorting dei nodi */

		switch (configuration.nodesorting){
			case SORT_N:
				printf("Sorting nodes w.r.t. owner. \n");
				qsort (mapping , counter_nodi, sizeof(int), prima0poi1);
				break;
			case SORT_O:
				printf("Sorting nodes w.r.t. owner, then w.r.t. outdegree\n");
				qsort (mapping , counter_nodi, sizeof(int), prima0poi1_outdeg);
				break;
			case SORT_I:
				printf("Sorting nodes w.r.t. owner, then w.r.t. indegree\n");
				qsort (mapping , counter_nodi, sizeof(int), prima0poi1_indeg);
				break;
			case SORT_OI:
				printf("Sorting nodes w.r.t. owner, then w.r.t. outdegree, then w.r.t. indegree\n");
				qsort (mapping , counter_nodi, sizeof(int), prima0poi1_outdeg_indeg);
				break;
			case SORT_A:
				printf("Sorting nodes w.r.t. owner, then w.r.t. outdegree+indegree\n");
				qsort (mapping , counter_nodi, sizeof(int), prima0poi1_alldeg);
				break;
		}
	


//		printf("PRIMA VERS:  "); for (idy=0; idy < counter_nodi; idy++) { printf("%d\t",mapping[idy]); } printf("\n");

		for (idy=0; idy < counter_nodi; idy++) {
			revmapping[mapping[idy]] = idy;
		}
//		printf("SECON VERS:  "); for (idy=0; idy < counter_nodi; idy++) { printf("%d\t",mapping[idy]); } printf("\n");
	}  // END GAMETYPE_PARITY || GAMETYPE_MPG ||...
	else {
		fprintf(stderr,"INPUT: CASO NON ANCORA IMPLEMENTATO\n");
	}
}










//host_csrPtrInSuccLists;
//host_csrPesiArchi;
//host_csrSuccLists;


void postparsing() {
	int idx;
	int idy;
	int idxInA = 0;
	int idxInB = 0;
	alloca_memoria_host();
	alloca_memoria_device();

// popola i vettori host_csr...[IntSorted] ricopiando da csr...[Ext]  e rinominando i nodi da 0 a N, in base al riordimanento determinato 
// nomeInt_of_nomeExt/nomeExt_of_nomeInt e da mapping[]
// oltre a riordinare/rinominare i nodi si devono coerentemente rinominare gli adiacenti (le destinazioni degli archi)
// OSS c'e' una doppia rinomina: nomeInt_of_nomeExt/nomeExt_of_nomeInt codificano la corrispondenza tra i nodi nel file (Ext) e
// una rinomina che li rinomina da 0 a N nell'ordine in cui sono incontrati nel file (Int==nomeOrdineLettura); 
// Tramite mapping/revmapping invece si tiene conto del riordino che pone prima i nodi che hanno owner=0 
// ( mapping[nomeInternoSorted] = nomeOrdineLettura  dove nomeInterno mette prima gli owner=0
// revmapping[nomeOrdineLettura] = nomeInternoSorted )

	for (idx=0; idx < counter_nodi0; idx++) {
		host_csrPtrInSuccLists[idxInA++] = idxInB;
		host_csrSuccLists[idxInB] = revmapping[nomeInt_of_nomeExt[csrSuccLists[csrPtrInSuccLists[mapping[idx]]]]];
		//printf("%d", revmapping[nomeInt_of_nomeExt[csrSuccLists[csrPtrInSuccLists[mapping[idx]]]]]); // almeno uno c'e'
		host_csrPesiArchi[idxInB++] = csrPesiArchi[csrPtrInSuccLists[mapping[idx]]];
		//printf("%d", csrPesiArchi[csrPtrInSuccLists[mapping[idx]]]); // almeno uno c'e'
		for (idy=(1+csrPtrInSuccLists[mapping[idx]]); idy < csrPtrInSuccLists[mapping[idx]+1]; idy++) { // gli altri
			host_csrSuccLists[idxInB] = revmapping[nomeInt_of_nomeExt[csrSuccLists[idy]]];
			//printf(",%d", revmapping[nomeInt_of_nomeExt[csrSuccLists[idy]]]);
			host_csrPesiArchi[idxInB++] = csrPesiArchi[idy];
			//printf(",%d", csrPesiArchi[idy]);
		}
		//printf("\n");
	}
	for (idx=counter_nodi0; idx < counter_nodi; idx++) {
		host_csrPtrInSuccLists[idxInA++] = idxInB;
		host_csrSuccLists[idxInB] = revmapping[nomeInt_of_nomeExt[csrSuccLists[csrPtrInSuccLists[mapping[idx]]]]];
		//printf("%d", revmapping[nomeInt_of_nomeExt[csrSuccLists[csrPtrInSuccLists[mapping[idx]]]]]); // almeno uno c'e'
		host_csrPesiArchi[idxInB++] = csrPesiArchi[csrPtrInSuccLists[mapping[idx]]];
		//printf("%d", csrPesiArchi[csrPtrInSuccLists[mapping[idx]]]); // almeno uno c'e'
		for (idy=(1+csrPtrInSuccLists[mapping[idx]]); idy < csrPtrInSuccLists[mapping[idx]+1]; idy++) { // gli altri
			host_csrSuccLists[idxInB] = revmapping[nomeInt_of_nomeExt[csrSuccLists[idy]]];
			//printf(",%d", revmapping[nomeInt_of_nomeExt[csrSuccLists[idy]]]);
			host_csrPesiArchi[idxInB++] = csrPesiArchi[idy];
			//printf(",%d", csrPesiArchi[idy]);
		}
		//printf("\n");
	}
	host_csrPtrInSuccLists[idxInA] = idxInB; // ultimo riferimento

	/* Determino MG_pesi come la somma su tutti i nodi n del max tra 0 e l'abs() del
	* peso minore (puo' essere negativo) tra gli archi uscenti da n,
	* Quindi per un n, se tutti gli archi uscenti da n hanno peso positivo, si ottiene 0 
	* E' una maggiorazione (generosa) del peso massimo (abs()) che potra' avere un 
	* qualsiasi ciclo negativo nel grafo */
	int maxdelnodo;
	if ((configuration.algoritmo == ALGOR_EG) || (configuration.algoritmo == ALGOR_EG0)) {
		MG_pesi = 0;
		for (idx=0; idx < counter_nodi; idx++) {
			maxdelnodo=0;
			for (idy=host_csrPtrInSuccLists[idx]; idy < host_csrPtrInSuccLists[idx+1]; idy++) {
				maxdelnodo = MAX(maxdelnodo,-(host_csrPesiArchi[idy]));
			}
			if (MG_pesi > (INT_MAX - maxdelnodo)) { //Check overflow
				// overflow handling
				sprintf(str,"%d + %d", maxdelnodo, MG_pesi);
				exitWithError("Error too high weights: %s --> overflow\n", str);
			} else {
				MG_pesi += maxdelnodo;
			}
		}
	}


	avg_outdegree = ((double)num_archi) / ((double)num_nodi);
	double sqsum = 0;
	for (idx=0; idx < counter_nodi; idx++) {
		sqsum += (((double)(csrPtrInSuccLists[idx+1]-csrPtrInSuccLists[idx]))-avg_outdegree) * (((double)(csrPtrInSuccLists[idx+1]-csrPtrInSuccLists[idx]))-avg_outdegree);
	}
	stddev_outdegree = sqrt(sqsum/(counter_nodi));


}




int main(int argc, char *argv[]) {
	struct timeb tp;
	double deltatime;
	int idz;

#if (defined(__CYGWIN__) || defined(__CYGWIN32__))
	char stringa[256] = { 'j', 'a', 'c', 'k', '\0' };
#elif defined(_WIN32)
	char stringa[256] = { 'b', 'i', 'l', 'l', '3', '2', '\0' };
#elif defined(_WIN64)
	char stringa[256] = { 'b', 'i', 'l', 'l', '6', '4', '\0' };
#elif defined(__APPLE__)
	char stringa[256] = { 'm', 'a', 'c', 'u', 's', 'e', 'r', '\0' };
#elif (defined(__linux__) || defined(__unix__))
	char stringa[256];
	cuserid(stringa);
#else 
	char stringa[256] = { 's', 't', 'r', 'a', 'n', 'g', 'e', 'r', '\0' };
#endif

	/* SET config param reading command-line options */	
	setconfig(argc, argv);
	setstat();
	
	//Calcolo soluzione
		printf("Process stats:\n\tUser: %s UID=%d\n", stringa, getuid());
		gethostname(stringa, 256);
		printf("\tProcess: PID=%d  running on %s\n", getpid(),stringa);
		ftime(&tp);
		printf("\tUnix time: %ld.%d  (%lu)\n", tp.time,tp.millitm, (ulong)clock());
		printf("\tCommand line: ");
		for (idz=0; idz<argc; idz++) { printf("%s ",argv[idz]); }
		printf("\n");fflush(stdout);
		ftime(&tp);
		deltatime = ((double)((long int)tp.time));
		deltatime += ((double)tp.millitm)/1000;

		switch (configuration.computationType){
			case GPU_COMPUTATION:
				if (configuration.deviceCount == 0) {
					printf("\nThere is no device supporting CUDA\n\n");
					exit(EXIT_FAILURE);
				} else {
					printf("CUDA stats:\n\t%d CUDA devices detected.\n", configuration.deviceCount);
					printf("\tSelect device number %d  name: %s\n", configuration.deviceid, configuration.devicename);
					printf("\tDevice compute capability %d.%d\n", configuration.capabilityMajor, configuration.capabilityMinor);
					printf("\tGPU clock: %.0f MHz (%0.2f GHz)\n", (float)(configuration.clockRate)* 1e-3f, (float)(configuration.clockRate)* 1e-6f);
					printf("\tDevice memory: %.0f MB\n", (float)(configuration.deviceProp_totalGlobalMem)/1048576.0f);
					printf("\tECC enabled: %s\n", (configuration.ECCEnabled)?"yes":"no");
					printf("\tShared memory size: %.0f KB\n", (float)(configuration.sharedMemPerBlock)/1024.0f);
					int driverVersion = 0, runtimeVersion = 0;
					cudaDriverGetVersion(&driverVersion);
					cudaRuntimeGetVersion(&runtimeVersion);
					printf("\tCUDA Driver Version / Runtime Version   %d.%d / %d.%d\n\n", driverVersion/1000, (driverVersion%100)/10, runtimeVersion/1000, (runtimeVersion%100)/10);
				}
				break;
 			case CPU_COMPUTATION:
				printf("CPU stats:\n\t ..TO DO..\n");
				break;
		}

		/* Parsing del output del grounder */
		printf("Parsing input (reading %s)...\n", (configuration.stdinputsource == 1)?"stdin":configuration.filename);fflush(stdout);
		parse_input();
		remap_instance();
		if (configuration.stdinputsource == 0) { gzclose(my_gzfile); }
		ftime(&tp);
		statistics.inputtime = (((double)((long int)tp.time))  + (((double)tp.millitm)/1000)) - deltatime;
		deltatime = ((double)((long int)tp.time));
		deltatime += ((double)tp.millitm)/1000;
		printf("Parsing completed (parsing time %lf).\n",statistics.inputtime);fflush(stdout);

		printf("\nStart post-parsing...\n");
		postparsing();

		ftime(&tp);
		deltatime = ((((double)((long int)tp.time))  + (((double)tp.millitm)/1000)) - deltatime);
		printf("Post-parsing completed (postparsing time %lf).\n",deltatime);fflush(stdout);

		printf("\nInput stats:\n\tNodes: %d (%d+%d)\n\tEdges: %d\n\tMax node Id: %d\n",num_nodi, counter_nodi0, counter_nodi-counter_nodi0, num_archi, max_nodo);
		printf("\tMin out-degree: %d\n", min_outdegree);
		printf("\tMax out-degree: %d\n\tMax weight(abs): %d\n", max_outdegree, max_pesi);
		printf("\tAvg out-degree: %.3lf\n\tStd dev: %.3lf\n\tRel std dev: %.2lf\n", avg_outdegree, stddev_outdegree, 100*stddev_outdegree/abs(avg_outdegree));fflush(stdout);
		printf("\tLow out-degree: 0: %d \t1: %d \t2: %d \t3: %d\n", numLowOutDegree[0], numLowOutDegree[1], numLowOutDegree[2], numLowOutDegree[3]);fflush(stdout);
		// spostato dopo traspose in CPU solver: printf("\tLow  in-degree: 0: %d \t1: %d \t2: %d \t3: %d\n", numLowInDegree[0], numLowInDegree[1], numLowInDegree[2], numLowInDegree[3]);

		/* RUN SOLVER */
		printf("\nPreparing to solve...\n");fflush(stdout);

		/* predispongo eventuale timeout subito prima di invocare il solver. Cosi' si trascura il tempo di input e preprocessing */
		timeout_expired = 0;
		if (configuration.timeoutOpt == SET_TIMEOUT_OPT) { 
			install_alarmhandler();
			printf("Setting %u seconds timeout... (not effective for some algorithms)\n",configuration.timeoutSeconds);
			alarm(configuration.timeoutSeconds);
		}
		fflush(stdout);

		switch (configuration.computationType){
			case CPU_COMPUTATION:
				cpu_solver();
				printf("------------------------\nTiming:\n");
				printf("Parsing time: %14.6lf sec \n", statistics.inputtime );
				printf("Postparsing time: %14.6lf sec \n", deltatime );
				//printf("Allocation time:   %14.6lf sec \n", statistics.alloctime/1000 );
				printf("Solving time: %14.6lf sec \n", statistics.solvingtime );
				printf("Total time: %14.6lf sec \n", deltatime+statistics.inputtime+ statistics.solvingtime);
				//printf("Total time:        %14.6lf sec \n", deltatime+statistics.inputtime+ (statistics.alloctime+statistics.solvingtime)/1000 );
				printf("------------------------\n");
				break;
 			case GPU_COMPUTATION:
				copia_dati_su_device();
				gpu_solver();
				printf("------------------------\nTiming:\n");
				printf("Parsing time: %14.6lf sec \n", statistics.inputtime );
				printf("Postparsing time: %14.6lf sec \n", deltatime );
				//printf("Allocation time:   %14.6lf sec \n", statistics.alloctime/1000 );
				printf("Solving time: %14.6lf sec \n", statistics.solvingtime/1000 );  //cuda usa diversa unita' di misura
				printf("Total time: %14.6lf sec \n", deltatime+statistics.inputtime+ (statistics.solvingtime)/1000 );
				printf("------------------------\n");
				break;
		}

		if (configuration.algoritmo != COMPARA) {
			if (configuration.onelineout != NO_OUTPUT) {
				printf("\nSolution:\n");fflush(stdout);
				if (configuration.onelineout == YES_ONELINEOUT) {
					output_solution_singleline();
				} else {
					output_solution_onenodeperline();
				}
			} else {
				printf("\nSolution output omitted.\n");fflush(stdout);
			}
		}
		dealloca_memoria_host();
		if (configuration.computationType == GPU_COMPUTATION) {
				dealloca_memoria_device();
				cudaDeviceReset();
		}
		ftime(&tp);
		printf("\nUnix time: %ld.%d  (%lu)\n", tp.time,tp.millitm, (ulong)clock());

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
	fprintf(stderr," EG solving: Reads an instance and applies the selected algorithm.\n");
	fprintf(stderr," Options:\n");
	fprintf(stderr,"  --help  -h  -?\n\tShow this message.\n");
	fprintf(stderr,"  --input FILENAME\n\tReads from FILENAME instead of using stdin. (Input can be in gzipped form. Default: stdin)\n");
	fprintf(stderr,"\n Options to assign/modify edge's -weights:\n");
	fprintf(stderr,"  --unit-weights\n\tAdd weights: w_i=1 for each node i. (Default: off)\n");
	fprintf(stderr,"  --exp-weights\n\tAdd weights: set w_i=(-num_nodes)^p_i. (Default: off)\n");
	fprintf(stderr,"  --rnd-weights [[L] U]\n\tAdd weights: w_i randomly chosen in [L..U]. (Default: off, L=-U, U=%d)\n",DEFAULT_ADD_RND_WEIGHTS_VAL);
	fprintf(stderr,"  --rnd-seed S\n\tUse S as seed for pseudo-random numbers. (Effective with --rnd-weights. Default: off)\n");
	fprintf(stderr,"\n Solving options:\n");
	fprintf(stderr,"  [--cpu|--gpu]\n\tChoose the computation type. Default: --cpu)\n");
	fprintf(stderr,"  --deviceid I\n\tSelection of the I-th CUDA capable device. (Effective with --gpu. Default: %d)\n", DEFAULT_DEVICE);
	fprintf(stderr,"  --tb T\n\tSet T=2^n as the number of threads-per-block. (Effective with --gpu. Default: T=%d)\n",DEFAULT_THREADSPERBLOCK);
	fprintf(stderr,"  --eg [N]\n\tUse basic implementation of EG algorithm (node driven). Performs at most N loops. (Default: on, N=|MG||V|)\n");
	fprintf(stderr,"  --eg0 [N]\n\tUse naive implementation of EG algorithm (node driven). Performs at most N loops. (Only effective with --cpu. Default: off, N=|MG||V|)\n");
	fprintf(stderr,"\n Useful weird options:\n");
	fprintf(stderr,"  --printdegrees\n\tPrint statistics about in/out degrees of nodes. (Only effective with --cpu. Default: off)\n");
	fprintf(stderr,"  --onelineout\n\tSolution in a single text line. (Default: off)\n");
	fprintf(stderr,"  --noout\n\tDo not print the solution explicitly. (Statistics are printed. Default: off)\n");
	fprintf(stderr,"  --timeout [S]\n\tStops after 1<S<%d sec of GPU solving time (approx.). (Not completely implemented. Default: off, S=%d)\n", INT_MAX, DEFAULT_TIMEOUT_SEC);
	fprintf(stderr,"\n Profiling options to sort nodes:\n");
	fprintf(stderr,"  --sort_n\n\tSorting w.r.t. owner (MAX<MIN). (Default)\n");
	fprintf(stderr,"  --sort_o\n\tSorting w.r.t. owner, then w.r.t. outdegree\n");
	fprintf(stderr,"  --sort_i\n\tSorting w.r.t. owner, then w.r.t. indegree\n");
	fprintf(stderr,"  --sort_oi\n\tSorting w.r.t. owner, then w.r.t. outdegree, then w.r.t. indegree\n");
	fprintf(stderr,"  --sort_a\n\tSorting w.r.t. owner, then w.r.t. outdegree+indegree\n");
	fprintf(stderr," In case of conflicting options, the last one is used.\n");
	fprintf(stderr,"\nExpected input format: Nodes are consecutive naturals up to MAXNODE.\n Input format:\n\t[ARENA] MAXNODE [NUMEDGES] ;\n\tnode  owner  node:weight, ..., node:weight  [\"string\"] ;\n \tnode  owner  node:weight, ..., node:weight  [\"string\"] ;\n\n");
	fflush(stderr);
}


//***********************************************//



/* Processa opzioni: */
void setconfig(int argc, char *argv[]) {

// SET DEFAULT PARAMS
	int i;
	struct timeb timeForRndSeed;


	configuration.computationType = DEFAULT_COMPUTATION_UNIT;
	configuration.algoritmo = DEFAULT_ALGOR;
// flag per memorizzare che input si legge (stdin o file)
	configuration.stdinputsource = 1;
	configuration.onelineout = NO_ONELINEOUT;
	configuration.print_degree_hist = DEFAULT_PRINT_DEGREEHIST;

	configuration.maxThreadsPerBlock = DEFAULT_THREADSPERBLOCK;

	configuration.add_weights_mode = DEFAULT_ADD_WEIGHTS;
	configuration.rndWeightLow = -DEFAULT_ADD_RND_WEIGHTS_VAL;
	configuration.rndWeightHigh = DEFAULT_ADD_RND_WEIGHTS_VAL;
	configuration.rndSeed = NO_USER_RND_SEED;

	configuration.deviceid = DEFAULT_DEVICE;
	configuration.threadsPerBlock = DEFAULT_THREADSPERBLOCK;

	configuration.max_loop_val = DEFAULT_MAX_LOOP_VAL;
	configuration.loop_slice = (1<<DEFAULT_LOOP_SLICE);
	configuration.loop_slice_for_EG = (1<<DEFAULT_LOOP_SLICE_FOR_EG);

	configuration.timeoutOpt = UNSET_TIMEOUT_OPT;
	configuration.timeoutSeconds = 0;

	configuration.nodesorting = DEFAULT_NODESORT;

	for( i = 1; i < argc; i++){
		if((strcmp(argv[i],"--help") == 0) || (strcmp(argv[i],"-?") == 0) || (strcmp(argv[i],"-h") == 0)) {
			printUsage(argv[0]);
			exit(EXIT_SUCCESS);
		}
		else if(strcmp(argv[i],"--cpu") == 0) {
			configuration.computationType = CPU_COMPUTATION;
		}
		else if(strcmp(argv[i],"--gpu") == 0) {
			configuration.computationType = GPU_COMPUTATION;
		}
		else if(strcmp(argv[i],"--printdegrees") == 0) {
			configuration.print_degree_hist = YES_PRINT_DEGREEHIST;
		}
		else if(strcmp(argv[i],"--onelineout") == 0) {
			configuration.onelineout = YES_ONELINEOUT;
		}
		else if(strcmp(argv[i],"--noout") == 0) {
			configuration.onelineout = NO_OUTPUT;
		}
		else if(strcmp(argv[i],"--deviceid") == 0) {
			checkExistsParameter(i+1, argc, "--deviceid", argv);
			configuration.deviceid = myatoi(argv[++i], "--deviceid", argv);  //atoi(argv[++i]);
			if (configuration.deviceid < 0) {
				fprintf(stderr,"\nERROR illegal parameter of option: --deviceid %s (specify a number identifying an available CUDA device)\n\n", argv[i]);
				printShortUsage(argv[0]);
				exit(1);
			}
		}
		else if(strcmp(argv[i],"--tb") == 0) {
			checkExistsParameter(i+1, argc, "--tb", argv);
			configuration.threadsPerBlock = myatoi(argv[++i], "--tb", argv);  //atoi(argv[++i]);
			if ((configuration.threadsPerBlock < 1) || (MY_HSTPOPLL(configuration.threadsPerBlock) != 1)) {
				fprintf(stderr,"\nERROR illegal parameter of option: --tb %s (specify a power of 2)\n\n", argv[i]);
				printShortUsage(argv[0]);
				exit(1);
			}
		}
		else if(strcmp(argv[i],"--input") == 0) {
			(configuration.filename)[0]='\0';
			checkExistsParameter(i+1, argc, "--input", argv);
			strcpy(configuration.filename,argv[++i]);
			// if ((inputsource=fopen(configuration.filename,"r")) == NULL) {
		//	if (my_gzfile != NULL) { gzclose(my_gzfile); } // test vero solo se in precedenza ho letto un altro "--input": chiude altro file
			configuration.stdinputsource = 0;
			if ((my_gzfile = gzopen(configuration.filename, "r")) == NULL) {
				fprintf(stderr,"\nERROR in opening file %s :", configuration.filename); perror(""); fprintf(stderr,"\n");
				printShortUsage(argv[0]);
				exit(1);
			} // else {
			// 	fprintf(stderr,"Reading from %s\n",configuration.filename);fflush(stderr);
			// 	my_gzfile = gzdopen(fileno(stdin), "rb");
			// 	if (my_gzfile == NULL) {
			// 		exitWithError("Cannot gzdopen stdin\n", "");
			// 	} else {
			// 		fprintf(stderr,"Reading from %s\n",configuration.filename);fflush(stderr);
			// 	}
			// }
		}
		else if(strcmp(argv[i],"--eg") == 0) {
			i++;
			if (checkExistsOptionalParameter(i, argc, argv) == 1) {
				int ttemp = myatoi(argv[i], "--eg", argv);  //atoi(argv[i]);
				if ((ttemp < 1) || (ttemp >= INT_MAX)) {
					fprintf(stderr,"\nERROR illegal parameter of option: --eg %s\n\n", argv[i]);
					printShortUsage(argv[0]);
					exit(1);
				}
				configuration.max_loop_val = ttemp;
			} else {
				i--;
				//configuration.max_loop_val = DEFAULT_MAX_LOOP_VAL;
			}
			configuration.algoritmo = ALGOR_EG;
		}
		else if(strcmp(argv[i],"--eg0") == 0) {
			i++;
			if (checkExistsOptionalParameter(i, argc, argv) == 1) {
				int ttemp = myatoi(argv[i], "--eg0", argv);  //atoi(argv[i]);
				if ((ttemp < 1) || (ttemp >= INT_MAX)) {
					fprintf(stderr,"\nERROR illegal parameter of option: --eg0 %s\n\n", argv[i]);
					printShortUsage(argv[0]);
					exit(1);
				}
				configuration.max_loop_val = ttemp;
			} else {
				i--;
				//configuration.max_loop_val = DEFAULT_MAX_LOOP_VAL;
			}
			configuration.algoritmo = ALGOR_EG0;
		}
		else if(strcmp(argv[i],"--timeout") == 0) {
			i++;
			if (checkExistsOptionalParameter(i, argc, argv) == 1) {
				int ttemp = myatoi(argv[i], "--timeout", argv);  //atoi(argv[i]);
				if ((ttemp <= 1) || (ttemp >= INT_MAX)) {
					fprintf(stderr,"\nERROR illegal parameter of option: --timeout %s\n\n", argv[i]);
					printShortUsage(argv[0]);
					exit(1);
				}
				configuration.timeoutSeconds = ttemp;
			} else {
				i--;
				//configuration.timeoutSeconds = DEFAULT_TIMEOUT_SEC;
			}
			configuration.timeoutOpt = SET_TIMEOUT_OPT;
		}
		else if(strcmp(argv[i],"--unit-weights") == 0) {
			configuration.add_weights_mode = ADD_UNIT_WEIGHTS;
		}
		else if(strcmp(argv[i],"--rnd-weights") == 0) {
			i++;
			if (checkExistsOptionalParameter(i, argc, argv) == 1) {
				int ttemp1 = myatoi(argv[i], "--rnd-weights", argv);  //atoi(argv[i]);
				i++;
				if (checkExistsOptionalParameter(i, argc, argv) == 1) {
					int ttemp2 = myatoi(argv[i], "--rnd-weights", argv);  //atoi(argv[i]);
					if ((ttemp2 < ttemp1) || (ttemp2 >= MAX_RAND_NUM) || (ttemp1 < -MAX_RAND_NUM)) {
						fprintf(stderr,"\nERROR illegal parameters of option: --rnd-weights %s %s\n\n", argv[i-1],argv[i]);
						printShortUsage(argv[0]);
						exit(1);
					} // due bound espliciti
					configuration.rndWeightLow = ttemp1;
					configuration.rndWeightHigh = ttemp2;
				} else { // un bound, intervallo simmetrico
					i--;
					if ((ttemp1 >= MAX_RAND_NUM) || (ttemp1 < 0)) {
						fprintf(stderr,"\nERROR illegal parameter of option: --rnd-weights %s\n\n", argv[i]);
						printShortUsage(argv[0]);
						exit(1);
					} // due bound espliciti
					configuration.rndWeightLow = -ttemp1;
					configuration.rndWeightHigh = ttemp1;
				}
			} else { // nessun bound specificato (uso i default assegnati prima)
				i--;
			}
			configuration.add_weights_mode = ADD_RND_WEIGHTS;
		}
		else if(strcmp(argv[i],"--rnd-seed") == 0) {
			checkExistsParameter(i+1, argc, "--rnd-seed", argv);
			configuration.rndSeed = myatoi(argv[++i], "--rnd-seed", argv);
			if ((configuration.rndSeed < 0) || (configuration.rndSeed >= INT_MAX)) {
				fprintf(stderr,"\nERROR illegal parameter of option: --rnd-seed %s)\n\n", argv[i]);
				printShortUsage(argv[0]);
				exit(1);
			}
		}
		else if(strcmp(argv[i],"--exp-weights") == 0) {
			configuration.add_weights_mode = ADD_EXP_WEIGHTS;
		}
		else if(strcmp(argv[i],"--no-weights") == 0) {
			configuration.add_weights_mode = NOT_ADD_WEIGHTS;
		}
		else if(strcmp(argv[i],"--sort_n") == 0) {
			configuration.nodesorting = SORT_N;
		}
		else if(strcmp(argv[i],"--sort_i") == 0) {
			configuration.nodesorting = SORT_I;
		}
		else if(strcmp(argv[i],"--sort_o") == 0) {
			configuration.nodesorting = SORT_O;
		}
		else if(strcmp(argv[i],"--sort_a") == 0) {
			configuration.nodesorting = SORT_A;
		}
		else if(strcmp(argv[i],"--sort_oi") == 0) {
			configuration.nodesorting = SORT_OI;
		}

		else
		{
			fprintf(stderr,"\nERROR unknown option: %s\n\n", argv[i]);
			printShortUsage(argv[0]);
			exit(1);
		}
	}

	int deviceCount = 0;
	if (configuration.computationType == GPU_COMPUTATION) {
		checkDevice(&deviceCount, &configuration); //sets: .devicename, .warpSize, .maxThreadsPerBlock, etc
		configuration.deviceCount = deviceCount;
		if (configuration.threadsPerBlock>configuration.maxThreadsPerBlock) {
			fprintf(stderr,"\nERROR illegal parameter of option: --tb (for device %d the number of threadsPerBlock must not exceed %d)\n\n", configuration.deviceid, configuration.maxThreadsPerBlock);
			printShortUsage(argv[0]);
			exit(1);
		}
	}

	if ( (configuration.computationType != CPU_COMPUTATION) ) { /* disattiva stampa degli histogrammi in/out degrees se non attive opzioni --cpu */
		configuration.print_degree_hist = NO_PRINT_DEGREEHIST;
	}

	if (configuration.rndSeed == NO_USER_RND_SEED) { // usa un seed generato usando system-time
		ftime(&timeForRndSeed);
		configuration.rndSeed = (int)(timeForRndSeed.millitm % SHRT_MAX) + (int)((timeForRndSeed.time/2) % INT_MAX);
	}  // altrimenti il default
	srand((uint) configuration.rndSeed);


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


//***********************************************//


//*******************************************//



void setstat() {
// SET DEFAULT VALUES
	statistics.solvingtime=0;
	statistics.alloctime=0;
	statistics.inputtime=0;
	statistics.device_usedGlobalMem=0;
}




//*******************************************//





