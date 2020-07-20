
/** \file cpu_solver.c
 * \brief  Codice host degli algoritmo del solver.
 *
 *
 */


#include "cpu_solver.h"
#include <omp.h>
extern char* str;
extern int counter_nodi;
extern int counter_nodi0;
//extern int counter_nodi0_1;
//extern int counter_nodi0_2;
extern int num_nodi;
extern int num_archi;
extern int max_pesi;
extern int MG_pesi;
extern char *nodeFlags;

extern int *host_csrPtrInSuccLists;
extern int *host_csrSuccLists;
extern int *host_csrPesiArchi;
extern int *host_cscPtrInPredLists;
extern int *host_cscPredLists;
extern int *host_cscPesiArchiPred;

extern int numLowInDegree[4];
extern int numAllInDegree[LEN_DEGREEHIST];
extern int numAllOutDegree[LEN_DEGREEHIST];

extern int *host_ResNodeValues1;
extern int *host_ResNodeValues2;
extern int *host_ResNodeValuesAux;

extern int *host_csrDataArchiAux;

extern uint timeout_expired;
extern config  configuration;
extern stat statistics;

void EG0_scpu_solver() {
        int idx;
        int idy;
        int val;
        long max_loop = configuration.max_loop_val;
        long loop;

        int *data1;
        int *data2;
        int *temp;
        int tempval;
        int flag1=0;

        printf("Initializing cpu EG0 solver.\n");

        max_loop = aggiorna_max_loop((long)num_archi, (long)num_nodi, (long)MG_pesi, max_loop);

        data1 = host_ResNodeValues1;
        data2 = host_ResNodeValues2;
        for (idx=0; idx<counter_nodi; idx++) {
                data1[idx] =0;
                data2[idx] =0;
        }
        printf("Running EG0 on CPU. (MG_pesi=%d max_loop=%ld num nodes=%d num archi=%d max weight=%d\n", MG_pesi, max_loop, num_nodi, num_archi, max_pesi); fflush(stdout);

        for (loop=1; loop<=max_loop; loop++) {
                flag1=0;
                for (idx=0; idx<counter_nodi; idx++) {
                        tempval = OMINUS(data1[host_csrSuccLists[host_csrPtrInSuccLists[idx]]] , host_csrPesiArchi[host_csrPtrInSuccLists[idx]]);
                        for (idy=(host_csrPtrInSuccLists[idx])+1; idy < host_csrPtrInSuccLists[idx+1]; idy++) {
                                val = OMINUS(data1[host_csrSuccLists[idy]] , host_csrPesiArchi[idy]);
                                if ((idx<counter_nodi0) && (tempval > val)) {
                                        tempval = val;
                                }
                                if ((idx>=counter_nodi0) && (tempval < val)) {
                                        tempval = val;
                                }
                        }
                        if (data2[idx] < tempval) {
                                flag1=1;
                                data2[idx] = tempval;
                        }
                }

                temp = data1; data1 = data2; data2 = temp; //swap ruoli degli array

                if (flag1 == 0) {break;}
                if (timeout_expired == 1) {break;}
        }
        if ((max_loop%2) != 0) { //se numero loop e' dispari, il risultato e' nell'array host_ResNodeValues2[]. Lo metto in host_ResNodeValues1[]
                temp = host_ResNodeValues1;
                host_ResNodeValues1 = host_ResNodeValues2;
                host_ResNodeValues2 = temp;
        }
        printf("End EG0 on CPU after %ld loops (each loop involves all nodes) (flag1=%d)\n", loop-1, flag1);
        statistics.processedNodes = ((long)(loop-1))*((long)num_nodi);
}


void EG0_cpu_solver() {
	int idx;
	int idy;
	long max_loop = configuration.max_loop_val;
	long loop;

	int *data1;
	int *data2;
	int *temp;
    // Varaible for OpenMP thread management
    int flag_per_thread[32];
    static int tid;
    static int nthreads=1;
    int c=0;
    for (c = 0; c < 32; c++) flag_per_thread[c] = 0;
#pragma omp threadprivate(tid)

#pragma omp parallel
{
        tid = omp_get_thread_num();
//        nthreads = omp_get_num_threads(); 
}

//	printf("Initializing cpu EG0 solver using (%d threads).\n", nthreads);

	max_loop = aggiorna_max_loop((long)num_archi, (long)num_nodi, (long)MG_pesi, max_loop);

	data1 = host_ResNodeValues1;
	data2 = host_ResNodeValues2;
	for (idx=0; idx<counter_nodi; idx++) {
		data1[idx] =0;
		data2[idx] =0;
	}
	printf("Running EG0 on CPU. (MG_pesi=%d max_loop=%ld num nodes=%d num archi=%d max weight=%d\n", MG_pesi, max_loop, num_nodi, num_archi, max_pesi); fflush(stdout);
        printf("counter nodi0 %d\n", counter_nodi0);
	for (loop=1; loop<=max_loop; loop++) {
        for (c = 0; c < 32; c++) flag_per_thread[c] = 0;
#pragma omp parallel for schedule(static, 8)
		for (idx=0; idx<counter_nodi; idx++) {
            printf("Tid loop%d\n", tid);
			int tempval = OMINUS(data1[host_csrSuccLists[host_csrPtrInSuccLists[idx]]] , host_csrPesiArchi[host_csrPtrInSuccLists[idx]]);
			for (idy=(host_csrPtrInSuccLists[idx])+1; idy < host_csrPtrInSuccLists[idx+1]; idy++) {
				int val = OMINUS(data1[host_csrSuccLists[idy]] , host_csrPesiArchi[idy]);
				if ((idx<counter_nodi0) && (tempval > val)) {
					tempval = val;
				}
				if ((idx>=counter_nodi0) && (tempval < val)) {
					tempval = val;
				}
			}
			if (data2[idx] < tempval) {
				flag_per_thread[tid]=1;
				data2[idx] = tempval;
			}
		}

		temp = data1; data1 = data2; data2 = temp; //swap ruoli degli array
                for (c = 0; c < 32; c++){ 
                    if (flag_per_thread[c] == 0){break;}
                }
		if (timeout_expired == 1) {break;}
	}
	if ((max_loop%2) != 0) { //se numero loop e' dispari, il risultato e' nell'array host_ResNodeValues2[]. Lo metto in host_ResNodeValues1[]
		temp = host_ResNodeValues1;
		host_ResNodeValues1 = host_ResNodeValues2;
		host_ResNodeValues2 = temp;
	}
	printf("End EG0 on CPU after %ld loops (each loop involves all nodes) (flag1=%d)\n", loop-1, flag1);
	statistics.processedNodes = ((long)(loop-1))*((long)num_nodi);
}




void EG_cpu_solver() {
	int idx;
	int idy;
	int val;
	long max_loop = configuration.max_loop_val;
	long loop;

	int *data1 = host_ResNodeValues1; // vettore dei risultati (f(v)) inizialmente gia' azzerato
	int temp;
	int flag1=0;
	int *stackL = host_csrDataArchiAux; //uso spazio host_csrDataArchiAux[] per memorizzare l'insieme L dei nodi con (potenziali) "inconsistenze"
						//inoltre uso il vettore di flags nodeFlags[] come bitmap per evitare di inserire doppi in stackL[]
						// N.B:: dato che ogni nodo ha in-degree e out-degree non nulli si ha  num_archi>=num_nodi
	int top_stackL = 0; // altezza dello stackL  //SCEGLIENDO questa si usa L come stack
	//int *queueL = stackL; // invece di stack uso queue  //SCEGLIENDO questa si usa L come queue
	int in_queueL = 0; 
	int out_queueL = 0; 
	int len_queueL = 0; 
	// Ora usa L come queue, le linee di codice per usare L come stack sono commentate.
	// Sperimentalmente, sembra che queue si comporti in media meglio di stack (per le istanze viste)



	printf("Initializing cpu EG  solver.\n");

	max_loop = aggiorna_max_loop((long)num_archi, (long)num_nodi, (long)MG_pesi, max_loop);

	// inizializza flags e stackL[] (= i nodi "inconsistenti")
	for (idx=0; idx<counter_nodi0; idx++) {
		temp=1; // finora sono tutti negativi
		idy=(host_csrPtrInSuccLists[idx]);
		while ((temp==1) && (idy < host_csrPtrInSuccLists[idx+1])) {
			if (host_csrPesiArchi[idy] >= 0) {
				temp = 0;
			}
			idy++;
		}
		if (temp==1) { // tutti outedges negativi
			nodeFlags[idx]=1;
			stackL[top_stackL++] = idx;
		}
	}

	for (idx=counter_nodi0; idx<counter_nodi; idx++) {
		temp=1; // finora sono nessun negativo
		idy=(host_csrPtrInSuccLists[idx]);
		while ((temp==1) && (idy < host_csrPtrInSuccLists[idx+1])) {
			if (host_csrPesiArchi[idy] < 0) {
				temp = 0;
			}
			idy++;
		}
		if (temp==0) { // almeno un outedge negativo
			nodeFlags[idx]=1;
			stackL[top_stackL++] = idx;
		}
	}
	in_queueL = top_stackL; 
	out_queueL = 0; 
	len_queueL = in_queueL-out_queueL; 

	host_csr2csc(num_nodi, num_nodi, num_archi, host_csrPesiArchi, host_csrSuccLists, host_csrPtrInSuccLists, host_cscPesiArchiPred, host_cscPredLists, host_cscPtrInPredLists);


	/* Calcolo gli in-degree  */
	{ int i,d;
		for (i=0; i<num_nodi; i++) {
			d = host_cscPtrInPredLists[i+1] - host_cscPtrInPredLists[i];
			if (d<(LEN_DEGREEHIST-1)) { (numAllInDegree[d])++;
			} else { (numAllInDegree[LEN_DEGREEHIST-1])++; }
			if (d<4) { (numLowInDegree[d])++; }
		}
		printf("\tLow  in-degree: 0: %d \t1: %d \t2: %d \t3: %d\n", numLowInDegree[0], numLowInDegree[1], numLowInDegree[2], numLowInDegree[3]);fflush(stdout);
	}
	if (configuration.print_degree_hist == YES_PRINT_DEGREEHIST) {
		int hist;
		printf("In  degrees:,%s,", (configuration.stdinputsource == 1)?"stdin":configuration.filename);
		for (hist=0; hist<(LEN_DEGREEHIST-1); hist++) {
			printf("%d,", numAllInDegree[hist]);
		}
		printf("%d\n", numAllInDegree[LEN_DEGREEHIST-1]);
		printf("Out degrees:,%s,", (configuration.stdinputsource == 1)?"stdin":configuration.filename);
		for (hist=0; hist<(LEN_DEGREEHIST-1); hist++) {
			printf("%d,", numAllOutDegree[hist]);
		}
		printf("%d\n", numAllOutDegree[LEN_DEGREEHIST-1]);
	}

	// Loop di calcolo dell'initial credit problem:
	printf("Running EG  on CPU. (MG_pesi=%d max_loop=%ld num nodes=%d num archi=%d max weight=%d\n", MG_pesi, max_loop, num_nodi, num_archi, max_pesi); fflush(stdout);

	loop=1;
	//while ((top_stackL>0) && (loop<=max_loop)) {
	while ((len_queueL>0) && (loop<=max_loop)) {

		// DUE modi (a seconda che L sia usato come stack o queue) per non selezionare gli elementi di L in ordine, ma pseudo-casualmente:
		//int r = rand()%top_stackL; int tt = stackL[r]; stackL[r] = stackL[top_stackL]; stackL[top_stackL] = tt;
		//int r = rand()%len_queueL; int tt = stackL[(out_queueL+r)%num_nodi]; stackL[(out_queueL+r)%num_nodi] = stackL[out_queueL]; stackL[out_queueL] = tt;
		// Solo per sperimentare. Ora DISATTIVATI.

		loop++;
		flag1=0;
		//idx = stackL[--top_stackL];  //preleva uno dei nodi da processare usando L come stack
		idx = stackL[out_queueL]; out_queueL=(out_queueL+1)%num_nodi;  //preleva uno dei nodi da processare usando L come queue
		len_queueL--;
		nodeFlags[idx]=0;   //riazzero per il ciclo successivo
	
		temp = OMINUS(data1[host_csrSuccLists[host_csrPtrInSuccLists[idx]]] , host_csrPesiArchi[host_csrPtrInSuccLists[idx]]);
		for (idy=(host_csrPtrInSuccLists[idx])+1; idy < host_csrPtrInSuccLists[idx+1]; idy++) {
			val = OMINUS(data1[host_csrSuccLists[idy]] , host_csrPesiArchi[idy]);
			if ((idx<counter_nodi0) && (temp > val)) {
				temp = val;
			}
			if ((idx>=counter_nodi0) && (temp < val)) {
				temp = val;
			}
		}


		// aggiunge predecessori del nodo aggiornato
		if (data1[idx] < temp) { // il valore aumenta
			//printf("  %d : %d --> %d)\n",idx,data1[idx],temp);
			data1[idx] = temp;
			flag1=1;
			// aggiugi PREDs in stackL GREZZO: aggiunge sempre!!! non usa counter come nell'articolo di Raffaella&C.
			int idz;
			for (idz=host_cscPtrInPredLists[idx]; idz<host_cscPtrInPredLists[idx+1]; idz++) {
				if (nodeFlags[host_cscPredLists[idz]] == 0) {
					//stackL[top_stackL++] = host_cscPredLists[idz]; // L come stack
					stackL[in_queueL] = host_cscPredLists[idz];  in_queueL=(in_queueL+1)%num_nodi;  // L come queue
					nodeFlags[host_cscPredLists[idz]]=1;
					len_queueL++; // L come queue
				}
			}
		}
		
		//if ((flag1 == 0) && (top_stackL==0)) {break;} // L come stack
		if ((flag1 == 0) && (len_queueL==0)) {break;}  // L come queue
		if (timeout_expired == 1) {break;}
	}

	printf("End EG  on CPU after %ld loops (each loop involves one node only) (flag1=%d)\n", loop-1, flag1);
	statistics.processedNodes = ((long)(loop-1));
}












void cpu_solver() {

	struct timeb tp;
	double deltacpusoltime;

	ftime(&tp);
	deltacpusoltime = ((double)((long int)tp.time));
	deltacpusoltime += ((double)tp.millitm)/1000;

	switch (configuration.algoritmo) {
		case ALGOR_EG0: // versione di EG naive (presa da master per facilitare multithread)
			EG0_cpu_solver();
			break;
		case ALGOR_EG:
			EG_cpu_solver();
			break;
		default:
			EG_cpu_solver();
			break;
	}
	ftime(&tp);
	statistics.solvingtime = ((((double)((long int)tp.time))  + (((double)tp.millitm)/1000)) - deltacpusoltime);
}

