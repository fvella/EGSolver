/** \file  dev_EG_alg.cu
 * \brief Header-file: Implementazione dell'algoritmo EG (derivato dagli Energy Games) 
 *
 * \todo Ottimizzare i kernel usando shared, __ldg, ...
 *
 */


#ifndef FLAGDEV_EG_ALG_CU
#include "csr2csc.cu"
#include "thrust_wrapper.cu"


extern int MG_pesi;
extern int *hdev_cscPtrInPredLists;
extern int *hdev_cscPredLists;
extern int *hdev_cscPesiArchiPred;

#define SLICESIZE 32
#define FIRSTOFFSET (SLICESIZE/2)

/* ************************************************************ */

/* operatore usato nell'algoritmo EG e definito nell'articolo "Faster algoritms for mean-payoff games".
 * E' definito nel file common.h.
 * Viene riportato anche qui solo per comodita' di lettura.
 * ( Riferisce la variabile globale MG_pesi )
 */
#ifndef OMINUS
        #define OMINUS(A,B) ( (((A)<INT_MAX) && (((A)-(B))<=(MG_pesi))) ? ((0<((A)-(B))) ? ((A)-(B)) : (0)) : (INT_MAX) )
#endif


#ifdef MYDEBUG
#define PRINTDEBUGSTUFF(A) { printf("PRINT-STUFF: %s:\n", (A)); \
			printf("index   "); for (randal=0; randal <num_nodi; randal++) {printf(" %3d",randal);} printf(" \n");  \
			CUDASAFE( cudaMemcpy(buffer, hdev_ResNodeValues1, num_nodi*sizeof(int),cudaMemcpyDeviceToHost) , "cudaMemcpyDeviceToHost hdev_ResNodeValues1");  \
			printf("Value1  ");for (randal=0; randal <num_nodi; randal++) {printf(" %3d",buffer[randal]);} printf(" \n");  \
			CUDASAFE( cudaMemcpy(buffer, hdev_nodeFlags1, num_nodi*sizeof(int),cudaMemcpyDeviceToHost) , "cudaMemcpyDeviceToHost hdev_nodeFlags1");  \
			printf("Flags1-1");for (randal=0; randal <num_nodi; randal++) {printf(" %3d",buffer[randal]-SHIFTNOME);} printf(" \n");  \
			CUDASAFE( cudaMemcpy(buffer, hdev_nodeFlags2, num_nodi*sizeof(int),cudaMemcpyDeviceToHost) , "cudaMemcpyDeviceToHost hdev_nodeFlags2");  \
			printf("Flags2-1");for (randal=0; randal <num_nodi; randal++) {printf(" %3d",buffer[randal]-SHIFTNOME);} printf(" \nAttivi:%d -----------------\n",numAttivi);  \
		   }
#else
#define PRINTDEBUGSTUFF(A) {;}
#endif 




/* SHIFTNOME: per poter usare 0 come valore reset dei flag devo slittare di +1 i nomi dei nodi/flag per non perdere il nodo 0 */
#define SHIFTNOME 1




/** \brief Calcolo su GPU dell'algoritmo EG node-based
 *
 * \details Implementazione node-based.
 * \n Un thread per nodo attivo. Ogni thread calcola il max (o min) dei valori relativi a tutti i suoi successori
 * con una scansione lineare.
 * Successivamente, qualora il valore ottenuto sia migliore di quello preesistente,
 * aggiorna tale valore e inserisce, con una scansione lineare, tutti i predecessori del nodo nell'insieme dei nodi attivi
 *
 * \n Questo kernel si deve alternare con il kernel gemello kernel_EG_all_global_2to1 che opera con i vettori dei flag scambiati
 * 
 **/
__global__ void kernel_EG_all_global_NEW1to2_none(const int num_0nodes, int num_nodi_attivi, const int MG_pesi) {
        int tidx = THREAD_ID;
        int temp, val, idy, old, nodo;
	int aux1, aux2, aux3, aux4, aux5;
	
        while (tidx < num_nodi_attivi) {
                nodo = dev_nodeFlags1[tidx] -SHIFTNOME;
		old = dev_ResNodeValues1[nodo];
		aux1 = dev_csrPtrInSuccLists[nodo];
		aux2 = dev_csrPtrInSuccLists[nodo+1];
		aux3 = dev_csrSuccLists[aux1];
		aux4 = dev_csrPesiArchi[aux1];
		aux5 = dev_ResNodeValues1[aux3];

                temp = OMINUS(aux5 , aux4);
		for (idy=aux1+1; idy < aux2; idy++) {
			aux3 = dev_csrSuccLists[idy];
			aux4 = dev_csrPesiArchi[idy];
			aux5 = dev_ResNodeValues1[aux3];
			val = OMINUS(aux5 , aux4);
                	if ((nodo<num_0nodes) && (temp > val)) { temp = val; }
                	if ((nodo>=num_0nodes) && (temp < val)) { temp = val; }
                }
        	if (old < temp) { // TODO:  li prendo tutti (da ottimizzare con "count")
        		dev_ResNodeValues1[nodo] = temp;
			aux1 = dev_cscPtrInPredLists[nodo];
			aux2 = dev_cscPtrInPredLists[nodo+1];
			for (idy=aux1; idy < aux2; idy++) {
				aux3 = dev_cscPredLists[idy];
				dev_nodeFlags2[aux3] = SHIFTNOME+aux3;  //RACE: qui diversi thread possono inserire lo stesso valore nella stessa posizione
			}
		}
		tidx += (blockDim.x * gridDim.x);
        }
}


/** \brief Calcolo su GPU dell'algoritmo EG node-based
 *
 * \n Vedi Kernel gemello di kernel_EG_all_global_1to2
 * 
 **/
__global__ void kernel_EG_all_global_NEW2to1_none(const int num_0nodes, int num_nodi_attivi, const int MG_pesi) {
        int tidx = THREAD_ID;
        int temp, val, idy, old, nodo;
	int aux1, aux2, aux3, aux4, aux5;
	
        while (tidx < num_nodi_attivi) {
                nodo = dev_nodeFlags2[tidx] -SHIFTNOME;
		old = dev_ResNodeValues1[nodo];
		aux1 = dev_csrPtrInSuccLists[nodo];
		aux2 = dev_csrPtrInSuccLists[nodo+1];
                //solo se non fatto dal chiamante con  memset  dev_nodeFlags2[tidx] = 0;
		aux3 = dev_csrSuccLists[aux1];
		aux4 = dev_csrPesiArchi[aux1];
		aux5 = dev_ResNodeValues1[aux3];

                temp = OMINUS(aux5 , aux4);
		for (idy=aux1+1; idy < aux2; idy++) {
			aux3 = dev_csrSuccLists[idy];
			aux4 = dev_csrPesiArchi[idy];
			aux5 = dev_ResNodeValues1[aux3];
			val = OMINUS(aux5 , aux4);
                	if ((nodo<num_0nodes) && (temp > val)) { 
                                        temp = val;
                        }
                	if ((nodo>=num_0nodes) && (temp < val)) {
                                        temp = val;
                        }
                }
        	if (old < temp) { // TODO:  li prendo tutti (da ottimizzare con "count")
        		dev_ResNodeValues1[nodo] = temp;
			aux1 = dev_cscPtrInPredLists[nodo];
			aux2 = dev_cscPtrInPredLists[nodo+1];
			for (idy=aux1; idy < aux2; idy++) {
				aux3 = dev_cscPredLists[idy];
				dev_nodeFlags1[aux3] = SHIFTNOME+aux3;  //RACE: qui diversi thread possono inserire lo stesso valore nella stessa posizione
			}
		}
		tidx += (blockDim.x * gridDim.x);
        }
}







/** \brief Calcolo su GPU dell'algoritmo EG node-based
 *
 * \details Implementazione node-based.
 * \n k (ora k=2) thread per nodo attivo. Ogni thread calcola il max (o min) dei valori relativi a 1/k dei suoi successori
 * con una scansione lineare. Poi con warp-shuffle i k thread si comunicano i risultati parziali.
 * Successivamente, qualora il valore ottenuto sia migliore di quello preesistente,
 * aggiorna tale valore e inserisce, con una scansione lineare, tutti i predecessori del nodo nell'insieme dei nodi attivi
 * Anche in questo caso i k thread processano 1/k dei predecessori.
 *
 * \n Questo kernel si deve alternare con il kernel gemello kernel_EG_all_global_2to1 che opera con i vettori dei flag scambiati
 * 
 **/
#define TRENTADUE 32

__global__ void kernel_EG_all_global_NEW1to2(const int num_0nodes, int num_nodi_attivi, const int MG_pesi) {
        int tidx = THREAD_ID;
        int temp, val, idy, old, nodo;
	int aux1, aux2, aux3, aux4, aux5;
	uint off = (uint)(tidx%TRENTADUE);
	
        while (tidx < num_nodi_attivi) {
                nodo = dev_nodeFlags1[tidx/TRENTADUE] -SHIFTNOME;
		old = dev_ResNodeValues1[nodo];
		aux1 = dev_csrPtrInSuccLists[nodo];
		aux2 = dev_csrPtrInSuccLists[nodo+1];
		aux3 = dev_csrSuccLists[aux1];
		aux4 = dev_csrPesiArchi[aux1];
		aux5 = dev_ResNodeValues1[aux3];

                temp = OMINUS(aux5 , aux4);
		for (idy=aux1+1+off; idy < aux2; idy+=TRENTADUE) {  // una parte di lavoro a testa per i thread con off= 0 ... TRENTADUE-1 
			aux3 = dev_csrSuccLists[idy];
			aux4 = dev_csrPesiArchi[idy];
			aux5 = dev_ResNodeValues1[aux3];
			val = OMINUS(aux5 , aux4);
                	if ((nodo<num_0nodes) && (temp > val)) { temp = val; }
                	if ((nodo>=num_0nodes) && (temp < val)) { temp = val; }
                }

		aux5 =__shfl_down(temp, 16, TRENTADUE); 
		if (off<16) { // 0,...,15 aggiornano il proprio temp se il caso
               		if ((nodo<num_0nodes) && (temp > aux5)) { temp = aux5; }
               		if ((nodo>=num_0nodes) && (temp < aux5)) { temp = aux5; }
		}
		aux5 =__shfl_down(temp, 8, TRENTADUE); 
		if (off<8) { // 0,...,7 aggiornano il proprio temp se il caso
               		if ((nodo<num_0nodes) && (temp > aux5)) { temp = aux5; }
               		if ((nodo>=num_0nodes) && (temp < aux5)) { temp = aux5; }
		}
		aux5 =__shfl_down(temp, 4, TRENTADUE); 
		if (off<4) { // 0,1,2,3 aggiornano il proprio temp se il caso
               		if ((nodo<num_0nodes) && (temp > aux5)) { temp = aux5; }
               		if ((nodo>=num_0nodes) && (temp < aux5)) { temp = aux5; }
		}
		aux5 =__shfl_down(temp, 2, TRENTADUE);  // i legge il temp  di i+2 (ha effetto per i=0,1)
		if (off<2) { // 0 e 1 aggiornano il proprio temp se il caso
               		if ((nodo<num_0nodes) && (temp > aux5)) { temp = aux5; } 
               		if ((nodo>=num_0nodes) && (temp < aux5)) { temp = aux5; }
		}
		aux5 =__shfl_down(temp, 1, TRENTADUE);  //0  legge il temp  di 1 (leggono anche gli altri ma e' ininfluente)
		if (off==0) { // 0 aggiorna il proprio temp se il caso
               		if ((nodo<num_0nodes) && (temp > aux5)) { temp = aux5; }
               		if ((nodo>=num_0nodes) && (temp < aux5)) { temp = aux5; }
		}
		aux5 =__shfl(temp, 0, TRENTADUE);  // tutti (0,...,31) leggono il temp  di 0

		//if (off==0) { // i due thread off=0 e off=1 sono nello stesso warp
               		if ((nodo<num_0nodes) && (temp > aux5)) { temp = aux5; }
               		if ((nodo>=num_0nodes) && (temp < aux5)) { temp = aux5; }
        		if (old < temp) { // TODO:  li prendo tutti (da ottimizzare con "count")
        			if (off==0) {dev_ResNodeValues1[nodo] = temp;}
				aux1 = dev_cscPtrInPredLists[nodo];
				aux2 = dev_cscPtrInPredLists[nodo+1];
				for (idy=aux1+off; idy < aux2; idy+=TRENTADUE) {
					aux3 = dev_cscPredLists[idy];
					dev_nodeFlags2[aux3] = SHIFTNOME+aux3;  //RACE: qui diversi thread possono inserire lo stesso valore nella stessa posizione
				}
			}
		//}
		tidx += (blockDim.x * gridDim.x);
	}
}


/** \brief Calcolo su GPU dell'algoritmo EG node-based
 *
 * \n Vedi Kernel gemello di kernel_EG_all_global_1to2
 * 
 **/
__global__ void kernel_EG_all_global_NEW2to1(const int num_0nodes, int num_nodi_attivi, const int MG_pesi) {
        int tidx = THREAD_ID;
        int temp, val, idy, old, nodo;
	int aux1, aux2, aux3, aux4, aux5;
	int off = tidx%TRENTADUE;
	
        while (tidx < num_nodi_attivi) {
                nodo = dev_nodeFlags2[tidx/TRENTADUE] -SHIFTNOME;
		old = dev_ResNodeValues1[nodo];
		aux1 = dev_csrPtrInSuccLists[nodo];
		aux2 = dev_csrPtrInSuccLists[nodo+1];
		aux3 = dev_csrSuccLists[aux1];
		aux4 = dev_csrPesiArchi[aux1];
		aux5 = dev_ResNodeValues1[aux3];

                temp = OMINUS(aux5 , aux4);
		for (idy=aux1+1+off; idy < aux2; idy+=TRENTADUE) {
			aux3 = dev_csrSuccLists[idy];
			aux4 = dev_csrPesiArchi[idy];
			aux5 = dev_ResNodeValues1[aux3];
			val = OMINUS(aux5 , aux4);
                	if ((nodo<num_0nodes) && (temp > val)) { temp = val; }
                	if ((nodo>=num_0nodes) && (temp < val)) { temp = val; }
                }
		
		aux5 =__shfl_down(temp, 16, TRENTADUE); 
		if (off<16) { // 0,...,15 aggiornano il proprio temp se il caso
               		if ((nodo<num_0nodes) && (temp > aux5)) { temp = aux5; }
               		if ((nodo>=num_0nodes) && (temp < aux5)) { temp = aux5; }
		}
		aux5 =__shfl_down(temp, 8, TRENTADUE); 
		if (off<8) { // 0,...,7 aggiornano il proprio temp se il caso
               		if ((nodo<num_0nodes) && (temp > aux5)) { temp = aux5; }
               		if ((nodo>=num_0nodes) && (temp < aux5)) { temp = aux5; }
		}
		aux5 =__shfl_down(temp, 4, TRENTADUE); 
		if (off<4) { // 0,1,2,3 aggiornano il proprio temp se il caso
               		if ((nodo<num_0nodes) && (temp > aux5)) { temp = aux5; }
               		if ((nodo>=num_0nodes) && (temp < aux5)) { temp = aux5; }
		}
		aux5 =__shfl_down(temp, 2, TRENTADUE);  // i legge il temp  di i+2 (ha effetto per i=0,1)
		if (off<2) { // 0 e 1 aggiornano il proprio temp se il caso
               		if ((nodo<num_0nodes) && (temp > aux5)) { temp = aux5; } 
               		if ((nodo>=num_0nodes) && (temp < aux5)) { temp = aux5; }
		}
		aux5 =__shfl_down(temp, 1, TRENTADUE);  //0  legge il temp  di 1 (leggono anche gli altri ma e' ininfluente)
		if (off==0) { // 0 aggiorna il proprio temp se il caso
               		if ((nodo<num_0nodes) && (temp > aux5)) { temp = aux5; }
               		if ((nodo>=num_0nodes) && (temp < aux5)) { temp = aux5; }
		}
		aux5 =__shfl(temp, 0, TRENTADUE);  // tutti leggono il temp  di 0

		//if (off==0) { // i due thread off=0 e off=1 sono nello stesso warp
               		if ((nodo<num_0nodes) && (temp > aux5)) { temp = aux5; }
               		if ((nodo>=num_0nodes) && (temp < aux5)) { temp = aux5; }
        		if (old < temp) { // TODO:  li prendo tutti (da ottimizzare con "count")
        			if (off==0) {dev_ResNodeValues1[nodo] = temp;}
				aux1 = dev_cscPtrInPredLists[nodo];
				aux2 = dev_cscPtrInPredLists[nodo+1];
				for (idy=aux1+off; idy < aux2; idy+=TRENTADUE) {
					aux3 = dev_cscPredLists[idy];
					dev_nodeFlags1[aux3] = SHIFTNOME+aux3;  //RACE: qui diversi thread possono inserire lo stesso valore nella stessa posizione
				}
			}
		//}
		tidx += (blockDim.x * gridDim.x);
        }
}


/** \brief Calcolo su GPU dell'algoritmo EG node-based
 *
 * \n Inizializza i dati per kernel_EG_all_global_1to2 e kernel_EG_all_global_2to1
 * 
 **/
__global__ void kernel_EG_initialize(const int num_0nodes, int num_nodi, const int MG_pesi) {
        int tidx = THREAD_ID;
        int temp, idy;

        while (tidx < num_nodi) { //un thread per ogni nodo
                temp = 1;
		idy=(dev_csrPtrInSuccLists[tidx]);
                if (tidx<num_0nodes) {
			while ((temp==1) && (idy < dev_csrPtrInSuccLists[tidx+1])) {
				if (dev_csrPesiArchi[idy] >= 0) {
					temp = 0;
				}
				idy++;
			}
			// set se tutti outedges negativi altrimenti 0
			dev_nodeFlags1[tidx]= ((temp==1) ? (SHIFTNOME+tidx) : 0);
                } else {
			while ((temp==1) && (idy < dev_csrPtrInSuccLists[tidx+1])) {
				if (dev_csrPesiArchi[idy] < 0) {
					temp = 0;
				}
				idy++;
			}
			// set se almeno un outedge negativo altrimenti 0
			dev_nodeFlags1[tidx]= ((temp==0) ? (SHIFTNOME+tidx) : 0);
                }
		tidx += (blockDim.x * gridDim.x);
        }
}








/** \brief Traspone la matrice di adiacenza
 *
 * \details Usa wrapper alla libreria CUSPARSE
 *
 **/
void EG_gpu_traspose_graph() {

        init_library();
        init_converter();

//MEMO: csr2csc(numrow,   numcol,   nnz,       dev_csrValA,       dev_csrRowPtrA,         dev_csrColIndA,    dev_cscValA,           dev_cscRowIndA,    dev_cscColPtrA);
        csr2csc(num_nodi, num_nodi, num_archi, hdev_csrPesiArchi, hdev_csrPtrInSuccLists, hdev_csrSuccLists, hdev_cscPesiArchiPred, hdev_cscPredLists, hdev_cscPtrInPredLists);
//TEST        testresult(num_nodi, num_nodi, num_archi, hdev_csrPesiArchi, hdev_csrPtrInSuccLists, hdev_csrSuccLists, hdev_cscPesiArchiPred, hdev_cscPredLists, hdev_cscPtrInPredLists,
//TEST                                                  host_csrPesiArchi, host_csrPtrInSuccLists, host_csrSuccLists, host_cscPesiArchiPred, host_cscPredLists, host_cscPtrInPredLists);

	cudaDeviceSynchronize(); // usa cusparseScsr2csc() che e' asincrona
        exit_converter();
        exit_library();
}






/** \brief Calcolo usando varie implementazioni di EG su GPU
 *
 * \details ...
 *
 **/
void EG_gpu_solver() {

	long total_num_processed_nodes = 0;
	long max_loop = configuration.max_loop_val;
	long extloop;
	int numAttivi;

	int tpb = (num_nodi< (int)configuration.threadsPerBlock) ? MIN(configuration.threadsPerBlock , MAX(configuration.warpSize, MYCEILSTEP(num_nodi, configuration.warpSize))) : configuration.threadsPerBlock;
	int nbs = MIN(MAX_BLOCKPERKERNEL, MYCEIL(num_nodi, tpb));

#ifdef MYDEBUG
	int randal;
	int * buffer = (int*)malloc((1+MAX(num_nodi,num_archi))*sizeof(int));
	printf("nbs=%d tpb=%d)\n",nbs, tpb);fflush(stdout);
#endif



	max_loop = aggiorna_max_loop((long)num_archi, (long)num_nodi, (long)MG_pesi, max_loop);
        extloop=max_loop;


	printf("Transposing arena...\t");fflush(stdout);
	EG_gpu_traspose_graph();
	printf("...done\n");fflush(stdout);


	if (configuration.loop_slice_for_EG != 1) {printf("WARNING: configuration.loop_slice_for_EG=%d  ignored!\n",configuration.loop_slice_for_EG);fflush(stdout);}
	else {
		printf("Running EG on GPU. (no shuffling or 32-shuffing depending on threshold=%d%%) (MG_pesi=%d max_loop=%ld extloop=%ld slice=%d num nodes=%d max weight=%d tpb=%d)\n", configuration.shuffleThreshold, MG_pesi, max_loop, extloop, configuration.loop_slice_for_EG, num_nodi, max_pesi, tpb); fflush(stdout);


		numAttivi = num_nodi;
//		printf("attivi=%d\tnbs=%d tpb=%d  counter_nodi0:%d num_nodi:%d MG_pesi:%d)\n",numAttivi,nbs, tpb, counter_nodi0, num_nodi, MG_pesi);fflush(stdout);
		kernel_EG_initialize<<<nbs, tpb>>>(counter_nodi0, num_nodi, MG_pesi);
		DEVSYNCANDCHECK_KER("kernel_EG_initialize  done")
		PRINTDEBUGSTUFF("dopo kernel_EG_initialize")

		remove_nulls(hdev_nodeFlags1, num_nodi, &numAttivi);
		PRINTDEBUGSTUFF("dopo remove_nulls iniziale")
                //printf("attivi=%d extloop=%d)\n",numAttivi,extloop);fflush(stdout);
                total_num_processed_nodes += (long)numAttivi;

		while ((extloop>0) && (numAttivi>0)) {
			tpb = ((TRENTADUE*numAttivi)< (int)configuration.threadsPerBlock) ? MIN(configuration.threadsPerBlock , MAX(configuration.warpSize, MYCEILSTEP(TRENTADUE*numAttivi, configuration.warpSize))) : configuration.threadsPerBlock;
			nbs = MIN(MAX_BLOCKPERKERNEL, MYCEIL(TRENTADUE*numAttivi, tpb));

//			printf("attivi=%d\tnbs=%d tpb=%d  counter_nodi0:%d num_nodi:%d MG_pesi:%d)\n",numAttivi,nbs, tpb, counter_nodi0, num_nodi, MG_pesi);fflush(stdout);
			if (((float) configuration.shuffleThreshold) < (100*((float)numAttivi)/((float)num_nodi))) {  // se percentuale di thread attivi elevata usa vertex-par
//	printf("1to2 perc attivi=%f run NONE)\n", (100*((float)numAttivi)/((float)num_nodi)) );fflush(stdout);
				kernel_EG_all_global_NEW1to2_none<<<nbs, tpb>>>(counter_nodi0, numAttivi, MG_pesi);
				DEVSYNCANDCHECK_KER("kernel_EG_all_global (no shuffle) (1) done")
				PRINTDEBUGSTUFF("dopo kernel_EG_all_global_NEW1to2_none")
			} else { // altrimenti shuffling con 32 thread per nodo
//	printf("\t\t\t1to2 perc attivi=%f run SH32)\n", (100*((float)numAttivi)/((float)num_nodi)) );fflush(stdout);
				kernel_EG_all_global_NEW1to2<<<nbs, tpb>>>(counter_nodi0, TRENTADUE*numAttivi, MG_pesi);
				DEVSYNCANDCHECK_KER("kernel_EG_all_global (shuffle-32) (1) done")
				PRINTDEBUGSTUFF("dopo kernel_EG_all_global_NEW1to2")
			}

			remove_nulls(hdev_nodeFlags2, num_nodi, &numAttivi);
			PRINTDEBUGSTUFF("dopo remove_nulls nodeFlags2")

			CUDASAFE( cudaMemset(hdev_nodeFlags1, 0, num_nodi*sizeof(hdev_nodeFlags1[0])) , "cudaMemset hdev_nodeFlags1");
			PRINTDEBUGSTUFF("dopo cudaMemset  nodeFlags1")

                        //printf("attivi=%d extloop=%d)\n",numAttivi,extloop);fflush(stdout);
                        total_num_processed_nodes += (long)numAttivi;
			extloop--;
			if (numAttivi < 1) {break;}  // caso in cui la computazione termina in un numero dispari di fasi

			tpb = ((TRENTADUE*numAttivi)< (int)configuration.threadsPerBlock) ? MIN(configuration.threadsPerBlock , MAX(configuration.warpSize, MYCEILSTEP(TRENTADUE*numAttivi, configuration.warpSize))) : configuration.threadsPerBlock;
			nbs = MIN(MAX_BLOCKPERKERNEL, MYCEIL(TRENTADUE*numAttivi, tpb));
//			printf("attivi=%d\tnbs=%d tpb=%d  counter_nodi0:%d num_nodi:%d MG_pesi:%d)\n",numAttivi,nbs, tpb, counter_nodi0, num_nodi, MG_pesi);fflush(stdout);
			if (((float) configuration.shuffleThreshold) < (100*((float)numAttivi)/((float)num_nodi))) {  // se percentuale di thread attivi elevata usa vertex-par
//	printf("2to1 perc attivi=%f run NONE)\n", (100*((float)numAttivi)/((float)num_nodi)) );fflush(stdout);
				kernel_EG_all_global_NEW2to1_none<<<nbs, tpb>>>(counter_nodi0, numAttivi, MG_pesi);
				DEVSYNCANDCHECK_KER("kernel_EG_all_global (no shuffle) (2) done")
				PRINTDEBUGSTUFF("dopo kernel_EG_all_global_NEW2to1_none")
			} else {
//	printf("\t\t\t2to1 perc attivi=%f run SH32)\n", (100*((float)numAttivi)/((float)num_nodi)) );fflush(stdout);
				kernel_EG_all_global_NEW2to1<<<nbs, tpb>>>(counter_nodi0, TRENTADUE*numAttivi, MG_pesi);
				DEVSYNCANDCHECK_KER("kernel_EG_all_global (shuffle-32) (2) done")
				PRINTDEBUGSTUFF("dopo kernel_EG_all_global_NEW2to1")
			}

			remove_nulls(hdev_nodeFlags1, num_nodi, &numAttivi);
			PRINTDEBUGSTUFF("dopo remove_nulls nodeFlags1")

			CUDASAFE( cudaMemset(hdev_nodeFlags2, 0, num_nodi*sizeof(hdev_nodeFlags2[0])) , "cudaMemset hdev_nodeFlags2");
			PRINTDEBUGSTUFF("dopo cudaMemset  nodeFlags2")

                        //printf("attivi=%d extloop=%d)\n",numAttivi,extloop);fflush(stdout);
                        total_num_processed_nodes += (long)numAttivi;
			extloop--;
//			printf("numAttivi: %d  residuo extloop=%ld\n",numAttivi,extloop);

			if (timeout_expired == 1) {break;}
		}
		printf("End EG on GPU after %ld loops (each loop involves one or more active nodes). Processed nodes %ld\n", max_loop-extloop, total_num_processed_nodes);
		statistics.processedNodes = total_num_processed_nodes;
	}
//	cudaDeviceSynchronize();
}






#define FLAGDEV_EG_ALG_CU 1
#endif


