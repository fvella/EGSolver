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



/* MACRO PER CALCOLO thread-per-block   *********************** */

#define EVAL_TPB_PRAGMA(SHUFFLE,PRAGMA,ELEMENTI) ( (((SHUFFLE)*(MYCEIL((ELEMENTI),(PRAGMA))))< (int)configuration.threadsPerBlock) ? MIN(configuration.threadsPerBlock , MAX(configuration.warpSize, MYCEILSTEP((SHUFFLE)*(MYCEIL((ELEMENTI),(PRAGMA))), configuration.warpSize))) : configuration.threadsPerBlock )

#define EVAL_TPB(SHUFFLE,ELEMENTI) ( (((SHUFFLE)*(ELEMENTI))                < (int)configuration.threadsPerBlock) ? MIN(configuration.threadsPerBlock , MAX(configuration.warpSize, MYCEILSTEP((SHUFFLE)*(ELEMENTI),                 configuration.warpSize))) : configuration.threadsPerBlock )



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
                        printf("Flags2-1");for (randal=0; randal <num_nodi; randal++) {printf(" %3d",buffer[randal]-SHIFTNOME);} printf("  -----------------\n");  \
                   }
#else
#define PRINTDEBUGSTUFF(A) {;}
#endif



/* SHIFTNOME: per poter usare 0 come valore reset dei flag devo slittare di +1 i nomi dei nodi/flag per non perdere il nodo 0 */
#define SHIFTNOME 1




/** \brief Calcolo su GPU dell'algoritmo EG node-based vertex-parallelism
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
__global__ void kernel_EG_all_global_NEW1to2_none(const int first, const int num_0nodes, int num_nodi_attivi, const int MG_pesi) {
	int tidx = THREAD_ID;
	int temp, val, idy, old, nodo;
	int aux1, aux2, aux3, aux4, aux5;
	
	while (NPRAG*tidx <= num_nodi_attivi) { //var  num_nodi_attivi  e' nodi
	    #pragma unroll
	    for(int k = 0; k < NPRAG; k++) {
		if (((tidx*NPRAG) + k) >= num_nodi_attivi){break;}
		nodo = dev_nodeFlags1[((tidx*NPRAG) + k)+first] -SHIFTNOME;

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
				//dev_nodeFlags2[aux3] = SHIFTNOME+aux3;  //RACE: qui diversi thread possono inserire lo stesso valore nella stessa posizione
				atomicCAS(dev_nodeFlags2+aux3, 0, SHIFTNOME+aux3); 
			}
		}
	    }
		tidx += ((blockDim.x * gridDim.x)*NPRAG);
	}
}


/** \brief Calcolo su GPU dell'algoritmo EG node-based vertex-parallelism
 *
 * \n Vedi Kernel gemello di kernel_EG_all_global_1to2
 * 
 **/
__global__ void kernel_EG_all_global_NEW2to1_none(const int first, const int num_0nodes, int num_nodi_attivi, const int MG_pesi) {
	int tidx = THREAD_ID;
	int temp, val, idy, old, nodo;
	int aux1, aux2, aux3, aux4, aux5;
	
	while (NPRAG*tidx <= num_nodi_attivi) { //var  num_nodi_attivi  e' nodi
	    #pragma unroll
	    for(int k = 0; k < NPRAG; k++) {
		if (((tidx*NPRAG) + k) >= num_nodi_attivi){break;}
		nodo = dev_nodeFlags2[((tidx*NPRAG) + k)+first] -SHIFTNOME;

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
			if ((nodo<num_0nodes) && (temp > val)) { temp = val; }
			if ((nodo>=num_0nodes) && (temp < val)) { temp = val; }
		}
		if (old < temp) { // TODO:  li prendo tutti (da ottimizzare con "count")
			dev_ResNodeValues1[nodo] = temp;
			aux1 = dev_cscPtrInPredLists[nodo];
			aux2 = dev_cscPtrInPredLists[nodo+1];
			for (idy=aux1; idy < aux2; idy++) {
				aux3 = dev_cscPredLists[idy];
				//dev_nodeFlags1[aux3] = SHIFTNOME+aux3;  //RACE: qui diversi thread possono inserire lo stesso valore nella stessa posizione
				atomicCAS(dev_nodeFlags1+aux3, 0, SHIFTNOME+aux3); 
			}
		}
	    }
		tidx += ((blockDim.x * gridDim.x)*NPRAG);
	}
}








/** \brief Inizializzazione su GPU dell'algoritmo EG node-based (fatto con vertex-parallelism (viene lanciato una volta sola))
 *
 * \n Inizializza i dati per kernel_EG_..._1to2 e kernel_EG_..._2to1
 * 
 **/
__global__ void kernel_EG_initialize(const int num_0nodes, int num_nodi, const int MG_pesi) {
	int tidx = THREAD_ID;
	int temp, idy;


	while (tidx < num_nodi) { //un thread per ogni nodo
		temp = 1;
		idy=(dev_csrPtrInSuccLists[tidx]);
		if (tidx<num_0nodes) {
			while (/*(temp==1) && */(idy < dev_csrPtrInSuccLists[tidx+1])) {
				if (dev_csrPesiArchi[idy] >= 0) {
					temp = 0;
				}
				idy++;
			}
			// set se tutti outedges negativi altrimenti 0
			dev_nodeFlags1[tidx]= ((temp==1) ? (SHIFTNOME+tidx) : 0);
		} else {
			while (/*(temp==1) && */(idy < dev_csrPtrInSuccLists[tidx+1])) {
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



/** \brief Inizializzazione su GPU dell'algoritmo EG con --outdegree
 *
 * \n Inizializza i dati per kernel_EG_..._1to2 e kernel_EG_..._2to1
 * 
 **/
__global__ void kernel_EG_initialize_split(const int shufflesplit_index, const int num_0nodes_1, const int num_0nodes_2, int num_nodi, const int MG_pesi) {
	int tidx = THREAD_ID;
	int temp, idy;


	while (tidx < num_nodi) { //un thread per ogni nodo
		temp = 1;
		idy=(dev_csrPtrInSuccLists[tidx]);
		if (tidx<num_0nodes_1) {
			while (/*(temp==1) &&*/ (idy < dev_csrPtrInSuccLists[tidx+1])) {
				if (dev_csrPesiArchi[idy] >= 0) {
					temp = 0;
				}
				idy++;
			}
			// set se tutti outedges negativi altrimenti 0
			dev_nodeFlags1[tidx]= ((temp==1) ? (SHIFTNOME+tidx) : 0);
		} else if (tidx<shufflesplit_index) {
			while (/*(temp==1) && */(idy < dev_csrPtrInSuccLists[tidx+1])) {
				if (dev_csrPesiArchi[idy] < 0) {
					temp = 0;
				}
				idy++;
			}
			// set se almeno un outedge negativo altrimenti 0
			dev_nodeFlags1[tidx]= ((temp==0) ? (SHIFTNOME+tidx) : 0);
		} else if (tidx<(shufflesplit_index+num_0nodes_2)) {
			while (/*(temp==1) && */(idy < dev_csrPtrInSuccLists[tidx+1])) {
				if (dev_csrPesiArchi[idy] >= 0) {
					temp = 0;
				}
				idy++;
			}
			// set se tutti outedges negativi altrimenti 0
			dev_nodeFlags1[tidx]= ((temp==1) ? (SHIFTNOME+tidx) : 0);
		} else {
			while (/*(temp==1) && */(idy < dev_csrPtrInSuccLists[tidx+1])) {
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

//MEMO: csr2csc(numrow,   numcol,   nnz,       dev_csrValA,       dev_csrRowPtrA,	 dev_csrColIndA,    dev_cscValA,	   dev_cscRowIndA,    dev_cscColPtrA);
	csr2csc(num_nodi, num_nodi, num_archi, hdev_csrPesiArchi, hdev_csrPtrInSuccLists, hdev_csrSuccLists, hdev_cscPesiArchiPred, hdev_cscPredLists, hdev_cscPtrInPredLists);
//TEST	testresult(num_nodi, num_nodi, num_archi, hdev_csrPesiArchi, hdev_csrPtrInSuccLists, hdev_csrSuccLists, hdev_cscPesiArchiPred, hdev_cscPredLists, hdev_cscPtrInPredLists,
//TEST						  host_csrPesiArchi, host_csrPtrInSuccLists, host_csrSuccLists, host_cscPesiArchiPred, host_cscPredLists, host_cscPtrInPredLists);

	cudaDeviceSynchronize(); // usa cusparseScsr2csc() che e' asincrona
	exit_converter();
	exit_library();
}











/** \brief Calcolo usando varie implementazioni di EG su GPU con vertex-parallelism
 *
 * \details ...
 *
 **/
void EG_gpu_solver_1() {

	long total_num_processed_nodes = 0;
	long max_loop = configuration.max_loop_val;
	long extloop;
	int numAttivi;

	int tpb = EVAL_TPB(1,num_nodi);
	int nbs = MIN(MAX_BLOCKPERKERNEL, MYCEIL(num_nodi, tpb));

#ifdef MYDEBUG
	int randal;
	int * buffer = (int*)malloc((1+MAX(num_nodi,num_archi))*sizeof(int));
	printf("nbs=%d tpb=%d)\n",nbs, tpb);fflush(stdout);
	CUDASAFE( cudaMemset(hdev_nodeFlags1, 0, num_nodi*sizeof(hdev_nodeFlags1[0])) , "cudaMemset hdev_nodeFlags1");
	CUDASAFE( cudaMemset(hdev_nodeFlags2, 0, num_nodi*sizeof(hdev_nodeFlags2[0])) , "cudaMemset hdev_nodeFlags2");
	{int idx,idy; for (idx=0; idx<num_nodi; idx++) { printf("%d)\t%d\n", nomeExt_of_nomeInt[mapping[idx]], host_ResNodeValues1[idx]); 
						   for (idy=host_csrPtrInSuccLists[idx]; idy<host_csrPtrInSuccLists[idx+1]; idy++) {
							   printf("\t\t%d(%d)\n", nomeExt_of_nomeInt[mapping[host_csrSuccLists[idy]]],  host_csrPesiArchi[idy]);} }}
#endif




	max_loop = aggiorna_max_loop((long)num_archi, (long)num_nodi, (long)MG_pesi, max_loop);
	extloop=max_loop;


	printf("Transposing arena...\t");fflush(stdout);
	EG_gpu_traspose_graph();
	printf("...done\n");fflush(stdout);


	printf("Running EG on GPU. (dev_EG_alg_shfl_none.cu) (MG_pesi=%d max_loop=%ld extloop=%ld num nodes=%d max weight=%d tpb=%d)\n", MG_pesi, max_loop, extloop, num_nodi, max_pesi, tpb); fflush(stdout);

	numAttivi = num_nodi;
	kernel_EG_initialize<<<nbs, tpb>>>(counter_nodi0, num_nodi, MG_pesi);
	DEVSYNCANDCHECK_KER("kernel_EG_initialize  done")

	remove_nulls(hdev_nodeFlags1, num_nodi, &numAttivi);
	total_num_processed_nodes += (long)numAttivi;

	while ((extloop>0) && (numAttivi>0)) {
		tpb = EVAL_TPB_PRAGMA(1,NPRAG,numAttivi);
		nbs = MIN(MAX_BLOCKPERKERNEL, MYCEIL((MYCEIL(numAttivi,NPRAG)), tpb));

		kernel_EG_all_global_NEW1to2_none<<<nbs, tpb>>>(0, counter_nodi0, numAttivi, MG_pesi);
		DEVSYNCANDCHECK_KER("kernel_EG_all_global (1) done")

		remove_nulls(hdev_nodeFlags2, num_nodi, &numAttivi);

		CUDASAFE( cudaMemset(hdev_nodeFlags1, 0, num_nodi*sizeof(hdev_nodeFlags1[0])) , "cudaMemset hdev_nodeFlags1");

		total_num_processed_nodes += (long)numAttivi;
		extloop--;
		if (numAttivi < 1) {break;}  // caso in cui la computazione termina in un numero dispari di fasi

		tpb = EVAL_TPB_PRAGMA(1,NPRAG,numAttivi);
		nbs = MIN(MAX_BLOCKPERKERNEL, MYCEIL((MYCEIL(numAttivi,NPRAG)), tpb));
		kernel_EG_all_global_NEW2to1_none<<<nbs, tpb>>>(0, counter_nodi0, numAttivi, MG_pesi);
		DEVSYNCANDCHECK_KER("kernel_EG_all_global (2) done")

		remove_nulls(hdev_nodeFlags1, num_nodi, &numAttivi);

		CUDASAFE( cudaMemset(hdev_nodeFlags2, 0, num_nodi*sizeof(hdev_nodeFlags2[0])) , "cudaMemset hdev_nodeFlags2");

		total_num_processed_nodes += (long)numAttivi;
		extloop--;

		if (timeout_expired == 1) {break;}
	}
	printf("End EG on GPU after %ld loops (each loop involves one or more active nodes). Processed nodes %ld\n", max_loop-extloop, total_num_processed_nodes);
	statistics.processedNodes = total_num_processed_nodes;
//	cudaDeviceSynchronize();
}






/** \brief Calcolo su GPU shuffling 2 thread
 *
 * \details 
 *
 * \n Questo kernel si deve alternare con il kernel gemello che opera con i vettori dei flag scambiati
 * 
 **/
#define DUE 2

__global__ void kernel_EG_all_global_NEW1to2_2tpv(const int first, const int num_0nodes, int num_nodi_attivi, const int MG_pesi) {
	int tidx = THREAD_ID;
	int temp, val, idy, old, nodo;
	int aux1, aux2, aux3, aux4, aux5;
	uint off = (uint)(tidx%DUE);
	
	while (NPRAG*(tidx/DUE) <= num_nodi_attivi) { //var  num_nodi_attivi  e' nodi
	    #pragma unroll
	    for(int k = 0; k < NPRAG; k++) {
		if ((((tidx/DUE)*NPRAG) + k) >= num_nodi_attivi){break;}
		nodo = dev_nodeFlags1[(((tidx/DUE)*NPRAG) + k)+first] -SHIFTNOME;

		old = dev_ResNodeValues1[nodo];
		aux1 = dev_csrPtrInSuccLists[nodo];
		aux2 = dev_csrPtrInSuccLists[nodo+1];
		aux3 = dev_csrSuccLists[aux1];
		aux4 = dev_csrPesiArchi[aux1];
		aux5 = dev_ResNodeValues1[aux3];

		temp = OMINUS(aux5 , aux4);
		for (idy=aux1+1+off; idy < aux2; idy+=DUE) {  // meta' lavoro a testa tra i due thread con off=0 e off=1 
			aux3 = dev_csrSuccLists[idy];
			aux4 = dev_csrPesiArchi[idy];
			aux5 = dev_ResNodeValues1[aux3];
			val = OMINUS(aux5 , aux4);
			if ((nodo<num_0nodes) && (temp > val)) { temp = val; }
			if ((nodo>=num_0nodes) && (temp < val)) { temp = val; }
		}

		
		// VECCHIO:  aux5 = __shfl_sync(0xFFFFFFFF, temp, (tidx%32)+1-2*off);  //1-off
                // RIMPIAZZATO DA:
		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 1, DUE);  //0  legge il temp  di 1 
		if (off==0) { // 0 aggiorna il proprio temp se il caso
	       		if ((nodo<num_0nodes) && (temp > aux5)) { temp = aux5; } 
	       		if ((nodo>=num_0nodes) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_sync(0xFFFFFFFF, temp, 0, DUE);  // tutti (0,1) leggono il temp  di 0
		//if (off==0) { // i due thread off=0 e off=1 sono nello stesso warp
	       		if ((nodo<num_0nodes) && (temp > aux5)) { temp = aux5; }
	       		if ((nodo>=num_0nodes) && (temp < aux5)) { temp = aux5; }
			if (old < temp) { // TODO:  li prendo tutti (da ottimizzare con "count")
				if (off==0) {dev_ResNodeValues1[nodo] = temp;}
				aux1 = dev_cscPtrInPredLists[nodo];
				aux2 = dev_cscPtrInPredLists[nodo+1];
				for (idy=aux1+off; idy < aux2; idy+=DUE) {
					aux3 = dev_cscPredLists[idy];
					//dev_nodeFlags2[aux3] = SHIFTNOME+aux3;  //RACE: qui diversi thread possono inserire lo stesso valore nella stessa posizione
					atomicCAS(dev_nodeFlags2+aux3, 0, SHIFTNOME+aux3); 
				}
			}
	    }
		tidx += ((blockDim.x * gridDim.x)*NPRAG);
	}
}


/** \brief Calcolo su GPU shuffling 2 thread
 *
 * \details 
 *
 * \n Questo kernel si deve alternare con il kernel gemello che opera con i vettori dei flag scambiati
 * 
 **/
__global__ void kernel_EG_all_global_NEW2to1_2tpv(const int first, const int num_0nodes, int num_nodi_attivi, const int MG_pesi) {
	int tidx = THREAD_ID;
	int temp, val, idy, old, nodo;
	int aux1, aux2, aux3, aux4, aux5;
	uint off = (uint)tidx%DUE;
	
	while (NPRAG*(tidx/DUE) <= num_nodi_attivi) { //var  num_nodi_attivi  e' nodi
	    #pragma unroll
	    for(int k = 0; k < NPRAG; k++) {
		if ((((tidx/DUE)*NPRAG) + k) >= num_nodi_attivi){break;}
		nodo = dev_nodeFlags2[(((tidx/DUE)*NPRAG) + k)+first] -SHIFTNOME;

		old = dev_ResNodeValues1[nodo];
		aux1 = dev_csrPtrInSuccLists[nodo];
		aux2 = dev_csrPtrInSuccLists[nodo+1];
		aux3 = dev_csrSuccLists[aux1];
		aux4 = dev_csrPesiArchi[aux1];
		aux5 = dev_ResNodeValues1[aux3];

		temp = OMINUS(aux5 , aux4);
		for (idy=aux1+1+off; idy < aux2; idy+=DUE) {
			aux3 = dev_csrSuccLists[idy];
			aux4 = dev_csrPesiArchi[idy];
			aux5 = dev_ResNodeValues1[aux3];
			val = OMINUS(aux5 , aux4);
			if ((nodo<num_0nodes) && (temp > val)) { temp = val; }
			if ((nodo>=num_0nodes) && (temp < val)) { temp = val; }
		}
		
		// VECCHIO:  aux5 = __shfl_sync(0xFFFFFFFF, temp, (tidx%32)+1-2*off);  //1-off
                // RIMPIAZZATO DA:
		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 1, DUE);  //0  legge il temp  di 1 
		if (off==0) { // 0 aggiorna il proprio temp se il caso
	       		if ((nodo<num_0nodes) && (temp > aux5)) { temp = aux5; } 
	       		if ((nodo>=num_0nodes) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_sync(0xFFFFFFFF, temp, 0, DUE);  // tutti (0,1) leggono il temp  di 0
		//if (off==0) { // i due thread off=0 e off=1 sono nello stesso warp
	       		if ((nodo<num_0nodes) && (temp > aux5)) { temp = aux5; }
	       		if ((nodo>=num_0nodes) && (temp < aux5)) { temp = aux5; }
			if (old < temp) { // TODO:  li prendo tutti (da ottimizzare con "count")
				if (off==0) {dev_ResNodeValues1[nodo] = temp;}
				aux1 = dev_cscPtrInPredLists[nodo];
				aux2 = dev_cscPtrInPredLists[nodo+1];
				for (idy=aux1+off; idy < aux2; idy+=DUE) {
					aux3 = dev_cscPredLists[idy];
					//dev_nodeFlags1[aux3] = SHIFTNOME+aux3;  //RACE: qui diversi thread possono inserire lo stesso valore nella stessa posizione
					atomicCAS(dev_nodeFlags1+aux3, 0, SHIFTNOME+aux3); 
				}
			}
		//}
	    }
		tidx += ((blockDim.x * gridDim.x)*NPRAG);
	}
}



/** \brief Calcolo usando varie implementazioni di EG su GPU usando 2-shuffling
 *
 * \details ...
 *
 **/
void EG_gpu_solver_2() {

	long total_num_processed_nodes = 0;
	long max_loop = configuration.max_loop_val;
	long extloop;
	int numAttivi;

	int tpb = EVAL_TPB(1,num_nodi);
	int nbs = MIN(MAX_BLOCKPERKERNEL, MYCEIL(num_nodi, tpb));




	max_loop = aggiorna_max_loop((long)num_archi, (long)num_nodi, (long)MG_pesi, max_loop);
	extloop=max_loop;


	printf("Transposing arena...\t");fflush(stdout);
	EG_gpu_traspose_graph();
	printf("...done\n");fflush(stdout);


	printf("Running EG on GPU. (dev_EG_alg_shfl_full_2tpv.cu) (MG_pesi=%d max_loop=%ld extloop=%ld num nodes=%d max weight=%d tpb=%d)\n", MG_pesi, max_loop, extloop, num_nodi, max_pesi, tpb); fflush(stdout);

	numAttivi = num_nodi;
//printf("attivi=%d\tnbs=%d tpb=%d  counter_nodi0:%d num_nodi:%d MG_pesi:%d)\n",numAttivi,nbs, tpb, counter_nodi0, num_nodi, MG_pesi);fflush(stdout);
	kernel_EG_initialize<<<nbs, tpb>>>(counter_nodi0, num_nodi, MG_pesi);
	DEVSYNCANDCHECK_KER("kernel_EG_initialize  done")

	remove_nulls(hdev_nodeFlags1, num_nodi, &numAttivi);
//printf("attivi=%d (INIT)   extloop=%ld\n", numAttivi,extloop);fflush(stdout);
	       total_num_processed_nodes += (long)numAttivi;

	while ((extloop>0) && (numAttivi>0)) {
		tpb = EVAL_TPB_PRAGMA(DUE,NPRAG,numAttivi);
		nbs = MIN(MAX_BLOCKPERKERNEL, MYCEIL(DUE*(MYCEIL(numAttivi,NPRAG)), tpb));

		kernel_EG_all_global_NEW1to2_2tpv<<<nbs, tpb>>>(0, counter_nodi0, numAttivi, MG_pesi);
		DEVSYNCANDCHECK_KER("kernel_EG_all_global (1) done")

		remove_nulls(hdev_nodeFlags2, num_nodi, &numAttivi);

		CUDASAFE( cudaMemset(hdev_nodeFlags1, 0, num_nodi*sizeof(hdev_nodeFlags1[0])) , "cudaMemset hdev_nodeFlags1");

//printf("attivi=%d extloop=%ld)\n",numAttivi,extloop);fflush(stdout);
		total_num_processed_nodes += (long)numAttivi;
		extloop--;
		if (numAttivi < 1) {break;}  // caso in cui la computazione termina in un numero dispari di fasi

		tpb = EVAL_TPB_PRAGMA(DUE,NPRAG,numAttivi);
		nbs = MIN(MAX_BLOCKPERKERNEL, MYCEIL(DUE*(MYCEIL(numAttivi,NPRAG)), tpb));
//printf("attivi=%d\tnbs=%d tpb=%d  counter_nodi0:%d num_nodi:%d MG_pesi:%d)\n",numAttivi,nbs, tpb, counter_nodi0, num_nodi, MG_pesi);fflush(stdout);
		kernel_EG_all_global_NEW2to1_2tpv<<<nbs, tpb>>>(0, counter_nodi0, numAttivi, MG_pesi);
		DEVSYNCANDCHECK_KER("kernel_EG_all_global (2) done")

		remove_nulls(hdev_nodeFlags1, num_nodi, &numAttivi);

		CUDASAFE( cudaMemset(hdev_nodeFlags2, 0, num_nodi*sizeof(hdev_nodeFlags2[0])) , "cudaMemset hdev_nodeFlags2");

//printf("attivi=%d extloop=%ld)\n",numAttivi,extloop);fflush(stdout);
		total_num_processed_nodes += (long)numAttivi;
		extloop--;
//printf("numAttivi: %d  residuo extloop=%ld\n",numAttivi,extloop);

		if (timeout_expired == 1) {break;}
	}
	printf("End EG on GPU after %ld loops (each loop involves one or more active nodes). Processed nodes %ld\n", max_loop-extloop, total_num_processed_nodes);
	statistics.processedNodes = total_num_processed_nodes;
//	cudaDeviceSynchronize();
}







/** \brief Calcolo su GPU shuffling 4 thread
 *
 * \details 
 *
 * \n Questo kernel si deve alternare con il kernel gemello che opera con i vettori dei flag scambiati
 * 
 **/
#define QUATTRO 4

__global__ void kernel_EG_all_global_NEW1to2_4tpv(const int first, const int num_0nodes, int num_nodi_attivi, const int MG_pesi) {
	int tidx = THREAD_ID;
	int temp, val, idy, old, nodo;
	int aux1, aux2, aux3, aux4, aux5;
	uint off = (uint)(tidx%QUATTRO);
	
	while (NPRAG*(tidx/QUATTRO) <= num_nodi_attivi) { //var  num_nodi_attivi  e' nodi
	    #pragma unroll
	    for(int k = 0; k < NPRAG; k++) {
		if ((((tidx/QUATTRO)*NPRAG) + k) >= num_nodi_attivi){break;}
		nodo = dev_nodeFlags1[(((tidx/QUATTRO)*NPRAG) + k)+first] -SHIFTNOME;

		old = dev_ResNodeValues1[nodo];
		aux1 = dev_csrPtrInSuccLists[nodo];
		aux2 = dev_csrPtrInSuccLists[nodo+1];
		aux3 = dev_csrSuccLists[aux1];
		aux4 = dev_csrPesiArchi[aux1];
		aux5 = dev_ResNodeValues1[aux3];

		temp = OMINUS(aux5 , aux4);
		for (idy=aux1+1+off; idy < aux2; idy+=QUATTRO) {  // una parte di lavoro a testa per i thread con off= 0 ... QUATTRO-1 
			aux3 = dev_csrSuccLists[idy];
			aux4 = dev_csrPesiArchi[idy];
			aux5 = dev_ResNodeValues1[aux3];
			val = OMINUS(aux5 , aux4);
			if ((nodo<num_0nodes) && (temp > val)) { temp = val; }
			if ((nodo>=num_0nodes) && (temp < val)) { temp = val; }
		}

		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 2, QUATTRO); // i legge il temp  di i+2 (ha effetto per i=0,1)
		if (off<2) { // 0 e 1 aggiornano il proprio temp se il caso
	       		if ((nodo<num_0nodes) && (temp > aux5)) { temp = aux5; } 
	       		if ((nodo>=num_0nodes) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 1, QUATTRO);  //0  legge il temp  di 1 (leggono anche gli altri ma e' ininfluente)
		if (off==0) { // 0 aggiorna il proprio temp se il caso
	       		if ((nodo<num_0nodes) && (temp > aux5)) { temp = aux5; } 
	       		if ((nodo>=num_0nodes) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_sync(0xFFFFFFFF, temp, 0, QUATTRO);  // tutti (0,1,2,3) leggono il temp  di 0

		//if (off==0) { // i due thread off=0 e off=1 sono nello stesso warp
	       		if ((nodo<num_0nodes) && (temp > aux5)) { temp = aux5; }
	       		if ((nodo>=num_0nodes) && (temp < aux5)) { temp = aux5; }
			if (old < temp) { // TODO:  li prendo tutti (da ottimizzare con "count")
				if (off==0) {dev_ResNodeValues1[nodo] = temp;}
				aux1 = dev_cscPtrInPredLists[nodo];
				aux2 = dev_cscPtrInPredLists[nodo+1];
				for (idy=aux1+off; idy < aux2; idy+=QUATTRO) {
					aux3 = dev_cscPredLists[idy];
					//dev_nodeFlags2[aux3] = SHIFTNOME+aux3;  //RACE: qui diversi thread possono inserire lo stesso valore nella stessa posizione
					atomicCAS(dev_nodeFlags2+aux3, 0, SHIFTNOME+aux3); 
				}
			}
		//}
	    }
		tidx += ((blockDim.x * gridDim.x)*NPRAG);
	}
}


/** \brief Calcolo su GPU shuffling 4 thread
 *
 * \details 
 *
 * \n Questo kernel si deve alternare con il kernel gemello che opera con i vettori dei flag scambiati
 * 
 **/
__global__ void kernel_EG_all_global_NEW2to1_4tpv(const int first, const int num_0nodes, int num_nodi_attivi, const int MG_pesi) {
	int tidx = THREAD_ID;
	int temp, val, idy, old, nodo;
	int aux1, aux2, aux3, aux4, aux5;
	uint off = (uint)(tidx%QUATTRO);
	
	while (NPRAG*(tidx/QUATTRO) <= num_nodi_attivi) { //var  num_nodi_attivi  e' nodi
	    #pragma unroll
	    for(int k = 0; k < NPRAG; k++) {
		if ((((tidx/QUATTRO)*NPRAG) + k) >= num_nodi_attivi){break;}
		nodo = dev_nodeFlags2[(((tidx/QUATTRO)*NPRAG) + k)+first] -SHIFTNOME;

		old = dev_ResNodeValues1[nodo];
		aux1 = dev_csrPtrInSuccLists[nodo];
		aux2 = dev_csrPtrInSuccLists[nodo+1];
		aux3 = dev_csrSuccLists[aux1];
		aux4 = dev_csrPesiArchi[aux1];
		aux5 = dev_ResNodeValues1[aux3];

		temp = OMINUS(aux5 , aux4);
		for (idy=aux1+1+off; idy < aux2; idy+=QUATTRO) {
			aux3 = dev_csrSuccLists[idy];
			aux4 = dev_csrPesiArchi[idy];
			aux5 = dev_ResNodeValues1[aux3];
			val = OMINUS(aux5 , aux4);
			if ((nodo<num_0nodes) && (temp > val)) { temp = val; }
			if ((nodo>=num_0nodes) && (temp < val)) { temp = val; }
		}
		
		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 2, QUATTRO); // i legge il temp  di i+2 (ha effetto per i=0,1)
		if (off<2) { // 0 e 1 aggiornano il proprio temp se il caso
	       		if ((nodo<num_0nodes) && (temp > aux5)) { temp = aux5; } 
	       		if ((nodo>=num_0nodes) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 1, QUATTRO);  //0  legge il temp  di 1 (leggono anche gli altri ma e' ininfluente)
		if (off==0) { // 0 aggiorna il proprio temp se il caso
	       		if ((nodo<num_0nodes) && (temp > aux5)) { temp = aux5; } 
	       		if ((nodo>=num_0nodes) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_sync(0xFFFFFFFF, temp, 0, QUATTRO);  // tutti (0,1,2,3) leggono il temp  di 0

		//if (off==0) { // i due thread off=0 e off=1 sono nello stesso warp
	       		if ((nodo<num_0nodes) && (temp > aux5)) { temp = aux5; }
	       		if ((nodo>=num_0nodes) && (temp < aux5)) { temp = aux5; }
			if (old < temp) { // TODO:  li prendo tutti (da ottimizzare con "count")
				if (off==0) {dev_ResNodeValues1[nodo] = temp;}
				aux1 = dev_cscPtrInPredLists[nodo];
				aux2 = dev_cscPtrInPredLists[nodo+1];
				for (idy=aux1+off; idy < aux2; idy+=QUATTRO) {
					aux3 = dev_cscPredLists[idy];
					//dev_nodeFlags1[aux3] = SHIFTNOME+aux3;  //RACE: qui diversi thread possono inserire lo stesso valore nella stessa posizione
					atomicCAS(dev_nodeFlags1+aux3, 0, SHIFTNOME+aux3); 
				}
			}
		//}
	    }
		tidx += ((blockDim.x * gridDim.x)*NPRAG);
	}
}




/** \brief Calcolo usando varie implementazioni di EG su GPU usando 4-shuffling
 *
 * \details ...
 *
 **/
void EG_gpu_solver_4() {

	long total_num_processed_nodes = 0;
	long max_loop = configuration.max_loop_val;
	long extloop;
	int numAttivi;

	int tpb = EVAL_TPB(1,num_nodi);
	int nbs = MIN(MAX_BLOCKPERKERNEL, MYCEIL(num_nodi, tpb));



	max_loop = aggiorna_max_loop((long)num_archi, (long)num_nodi, (long)MG_pesi, max_loop);
	extloop=max_loop;


	printf("Transposing arena...\t");fflush(stdout);
	EG_gpu_traspose_graph();
	printf("...done\n");fflush(stdout);


	printf("Running EG on GPU. (dev_EG_alg_shfl_full_4tpv.cu) (MG_pesi=%d max_loop=%ld extloop=%ld num nodes=%d max weight=%d tpb=%d)\n", MG_pesi, max_loop, extloop, num_nodi, max_pesi, tpb); fflush(stdout);

	numAttivi = num_nodi;
//	printf("attivi=%d\tnbs=%d tpb=%d  counter_nodi0:%d num_nodi:%d MG_pesi:%d)\n",numAttivi,nbs, tpb, counter_nodi0, num_nodi, MG_pesi);fflush(stdout);
	kernel_EG_initialize<<<nbs, tpb>>>(counter_nodi0, num_nodi, MG_pesi);
	DEVSYNCANDCHECK_KER("kernel_EG_initialize  done")

	remove_nulls(hdev_nodeFlags1, num_nodi, &numAttivi);
	//printf("attivi=%d extloop=%d)\n",numAttivi,extloop);fflush(stdout);
	total_num_processed_nodes += (long)numAttivi;

	while ((extloop>0) && (numAttivi>0)) {
		tpb = EVAL_TPB_PRAGMA(QUATTRO,NPRAG,numAttivi);
		nbs = MIN(MAX_BLOCKPERKERNEL, MYCEIL(QUATTRO*(MYCEIL(numAttivi,NPRAG)), tpb));

//		printf("attivi=%d\tnbs=%d tpb=%d  counter_nodi0:%d num_nodi:%d MG_pesi:%d)\n",numAttivi,nbs, tpb, counter_nodi0, num_nodi, MG_pesi);fflush(stdout);
		kernel_EG_all_global_NEW1to2_4tpv<<<nbs, tpb>>>(0, counter_nodi0, numAttivi, MG_pesi);
		DEVSYNCANDCHECK_KER("kernel_EG_all_global (1) done")

		remove_nulls(hdev_nodeFlags2, num_nodi, &numAttivi);

		CUDASAFE( cudaMemset(hdev_nodeFlags1, 0, num_nodi*sizeof(hdev_nodeFlags1[0])) , "cudaMemset hdev_nodeFlags1");

		//printf("attivi=%d extloop=%d)\n",numAttivi,extloop);fflush(stdout);
		total_num_processed_nodes += (long)numAttivi;
		extloop--;
		if (numAttivi < 1) {break;}  // caso in cui la computazione termina in un numero dispari di fasi

		tpb = EVAL_TPB_PRAGMA(QUATTRO,NPRAG,numAttivi);
		nbs = MIN(MAX_BLOCKPERKERNEL, MYCEIL(QUATTRO*(MYCEIL(numAttivi,NPRAG)), tpb));
//		printf("attivi=%d\tnbs=%d tpb=%d  counter_nodi0:%d num_nodi:%d MG_pesi:%d)\n",numAttivi,nbs, tpb, counter_nodi0, num_nodi, MG_pesi);fflush(stdout);
		kernel_EG_all_global_NEW2to1_4tpv<<<nbs, tpb>>>(0, counter_nodi0, numAttivi, MG_pesi);
		DEVSYNCANDCHECK_KER("kernel_EG_all_global (2) done")

		remove_nulls(hdev_nodeFlags1, num_nodi, &numAttivi);

		CUDASAFE( cudaMemset(hdev_nodeFlags2, 0, num_nodi*sizeof(hdev_nodeFlags2[0])) , "cudaMemset hdev_nodeFlags2");

		//printf("attivi=%d extloop=%d)\n",numAttivi,extloop);fflush(stdout);
		total_num_processed_nodes += (long)numAttivi;
		extloop--;
//		printf("numAttivi: %d  residuo extloop=%ld\n",numAttivi,extloop);

		if (timeout_expired == 1) {break;}
	}
	printf("End EG on GPU after %ld loops (each loop involves one or more active nodes). Processed nodes %ld\n", max_loop-extloop, total_num_processed_nodes);
	statistics.processedNodes = total_num_processed_nodes;
//	cudaDeviceSynchronize();
}







/** \brief Calcolo su GPU shuffling 8 thread
 *
 * \details 
 *
 * \n Questo kernel si deve alternare con il kernel gemello che opera con i vettori dei flag scambiati
 * 
 **/
#define OTTO 8

__global__ void kernel_EG_all_global_NEW1to2_8tpv(const int first, const int num_0nodes, int num_nodi_attivi, const int MG_pesi) {
	int tidx = THREAD_ID;
	int temp, val, idy, old, nodo;
	int aux1, aux2, aux3, aux4, aux5;
	uint off = (uint)(tidx%OTTO);
	
	while (NPRAG*(tidx/OTTO) <= num_nodi_attivi) { //var  num_nodi_attivi  e' nodi
	    #pragma unroll
	    for(int k = 0; k < NPRAG; k++) {
		if ((((tidx/OTTO)*NPRAG) + k) >= num_nodi_attivi){break;}
		nodo = dev_nodeFlags1[(((tidx/OTTO)*NPRAG) + k)+first] -SHIFTNOME;

		old = dev_ResNodeValues1[nodo];
		aux1 = dev_csrPtrInSuccLists[nodo];
		aux2 = dev_csrPtrInSuccLists[nodo+1];
		aux3 = dev_csrSuccLists[aux1];
		aux4 = dev_csrPesiArchi[aux1];
		aux5 = dev_ResNodeValues1[aux3];

		temp = OMINUS(aux5 , aux4);
		for (idy=aux1+1+off; idy < aux2; idy+=OTTO) {  // una parte di lavoro a testa per i thread con off= 0 ... OTTO-1 
			aux3 = dev_csrSuccLists[idy];
			aux4 = dev_csrPesiArchi[idy];
			aux5 = dev_ResNodeValues1[aux3];
			val = OMINUS(aux5 , aux4);
			if ((nodo<num_0nodes) && (temp > val)) { temp = val; }
			if ((nodo>=num_0nodes) && (temp < val)) { temp = val; }
		}

		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 4, OTTO); 
		if (off<4) { // 0,1,2,3 aggiornano il proprio temp se il caso
	       		if ((nodo<num_0nodes) && (temp > aux5)) { temp = aux5; }
	       		if ((nodo>=num_0nodes) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 2, OTTO);  // i legge il temp  di i+2 (ha effetto per i=0,1)
		if (off<2) { // 0 e 1 aggiornano il proprio temp se il caso
	       		if ((nodo<num_0nodes) && (temp > aux5)) { temp = aux5; } 
	       		if ((nodo>=num_0nodes) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 1, OTTO);  //0  legge il temp  di 1 (leggono anche gli altri ma e' ininfluente)
		if (off==0) { // 0 aggiorna il proprio temp se il caso
	       		if ((nodo<num_0nodes) && (temp > aux5)) { temp = aux5; }
	       		if ((nodo>=num_0nodes) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_sync(0xFFFFFFFF, temp, 0, OTTO);  // tutti (0,...,7) leggono il temp  di 0

		//if (off==0) { // i due thread off=0 e off=1 sono nello stesso warp
	       		if ((nodo<num_0nodes) && (temp > aux5)) { temp = aux5; }
	       		if ((nodo>=num_0nodes) && (temp < aux5)) { temp = aux5; }
			if (old < temp) { // TODO:  li prendo tutti (da ottimizzare con "count")
				if (off==0) {dev_ResNodeValues1[nodo] = temp;}
				aux1 = dev_cscPtrInPredLists[nodo];
				aux2 = dev_cscPtrInPredLists[nodo+1];
				for (idy=aux1+off; idy < aux2; idy+=OTTO) {
					aux3 = dev_cscPredLists[idy];
					//dev_nodeFlags2[aux3] = SHIFTNOME+aux3;  //RACE: qui diversi thread possono inserire lo stesso valore nella stessa posizione
					atomicCAS(dev_nodeFlags2+aux3, 0, SHIFTNOME+aux3); 
				}
			}
		//}
	    }
		tidx += ((blockDim.x * gridDim.x)*NPRAG);
	}
}


/** \brief Calcolo su GPU shuffling 8 thread
 *
 * \details 
 *
 * \n Questo kernel si deve alternare con il kernel gemello che opera con i vettori dei flag scambiati
 * 
 **/
__global__ void kernel_EG_all_global_NEW2to1_8tpv(const int first, const int num_0nodes, int num_nodi_attivi, const int MG_pesi) {
	int tidx = THREAD_ID;
	int temp, val, idy, old, nodo;
	int aux1, aux2, aux3, aux4, aux5;
	uint off = (uint)(tidx%OTTO);
	
	while (NPRAG*(tidx/OTTO) <= num_nodi_attivi) { //var  num_nodi_attivi  e' nodi
	    #pragma unroll
	    for(int k = 0; k < NPRAG; k++) {
		if ((((tidx/OTTO)*NPRAG) + k) >= num_nodi_attivi){break;}
		nodo = dev_nodeFlags2[(((tidx/OTTO)*NPRAG) + k)+first] -SHIFTNOME;

		old = dev_ResNodeValues1[nodo];
		aux1 = dev_csrPtrInSuccLists[nodo];
		aux2 = dev_csrPtrInSuccLists[nodo+1];
		aux3 = dev_csrSuccLists[aux1];
		aux4 = dev_csrPesiArchi[aux1];
		aux5 = dev_ResNodeValues1[aux3];

		temp = OMINUS(aux5 , aux4);
		for (idy=aux1+1+off; idy < aux2; idy+=OTTO) {
			aux3 = dev_csrSuccLists[idy];
			aux4 = dev_csrPesiArchi[idy];
			aux5 = dev_ResNodeValues1[aux3];
			val = OMINUS(aux5 , aux4);
			if ((nodo<num_0nodes) && (temp > val)) { temp = val; }
			if ((nodo>=num_0nodes) && (temp < val)) { temp = val; }
		}
		
		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 4, OTTO); 
		if (off<4) { // 0,1,2,3 aggiornano il proprio temp se il caso
	       		if ((nodo<num_0nodes) && (temp > aux5)) { temp = aux5; }
	       		if ((nodo>=num_0nodes) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 2, OTTO);  // i legge il temp  di i+2 (ha effetto per i=0,1)
		if (off<2) { // 0 e 1 aggiornano il proprio temp se il caso
	       		if ((nodo<num_0nodes) && (temp > aux5)) { temp = aux5; } 
	       		if ((nodo>=num_0nodes) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 1, OTTO);  //0  legge il temp  di 1 (leggono anche gli altri ma e' ininfluente)
		if (off==0) { // 0 aggiorna il proprio temp se il caso
	       		if ((nodo<num_0nodes) && (temp > aux5)) { temp = aux5; }
	       		if ((nodo>=num_0nodes) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_sync(0xFFFFFFFF, temp, 0, OTTO);  // tutti (0,...,7) leggono il temp  di 0

		//if (off==0) { // i due thread off=0 e off=1 sono nello stesso warp
	       		if ((nodo<num_0nodes) && (temp > aux5)) { temp = aux5; }
	       		if ((nodo>=num_0nodes) && (temp < aux5)) { temp = aux5; }
			if (old < temp) { // TODO:  li prendo tutti (da ottimizzare con "count")
				if (off==0) {dev_ResNodeValues1[nodo] = temp;}
				aux1 = dev_cscPtrInPredLists[nodo];
				aux2 = dev_cscPtrInPredLists[nodo+1];
				for (idy=aux1+off; idy < aux2; idy+=OTTO) {
					aux3 = dev_cscPredLists[idy];
					//dev_nodeFlags1[aux3] = SHIFTNOME+aux3;  //RACE: qui diversi thread possono inserire lo stesso valore nella stessa posizione
					atomicCAS(dev_nodeFlags1+aux3, 0, SHIFTNOME+aux3); 
				}
			}
		//}
	    }
		tidx += ((blockDim.x * gridDim.x)*NPRAG);
	}
}





/** \brief Calcolo usando varie implementazioni di EG su GPU usando 8-shuffling
 *
 * \details ...
 *
 **/
void EG_gpu_solver_8() {

	long total_num_processed_nodes = 0;
	long max_loop = configuration.max_loop_val;
	long extloop;
	int numAttivi;

	int tpb = EVAL_TPB(1,num_nodi);
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


	printf("Running EG on GPU. (dev_EG_alg_shfl_full_8tpv.cu) (MG_pesi=%d max_loop=%ld extloop=%ld num nodes=%d max weight=%d tpb=%d)\n", MG_pesi, max_loop, extloop, num_nodi, max_pesi, tpb); fflush(stdout);

	numAttivi = num_nodi;
//	printf("attivi=%d\tnbs=%d tpb=%d  counter_nodi0:%d num_nodi:%d MG_pesi:%d)\n",numAttivi,nbs, tpb, counter_nodi0, num_nodi, MG_pesi);fflush(stdout);
	kernel_EG_initialize<<<nbs, tpb>>>(counter_nodi0, num_nodi, MG_pesi);
	DEVSYNCANDCHECK_KER("kernel_EG_initialize  done")

	remove_nulls(hdev_nodeFlags1, num_nodi, &numAttivi);
	//printf("attivi=%d extloop=%d)\n",numAttivi,extloop);fflush(stdout);
	total_num_processed_nodes += (long)numAttivi;

	while ((extloop>0) && (numAttivi>0)) {
		tpb = EVAL_TPB_PRAGMA(OTTO,NPRAG,numAttivi);
		nbs = MIN(MAX_BLOCKPERKERNEL, MYCEIL(OTTO*(MYCEIL(numAttivi,NPRAG)), tpb));

//		printf("attivi=%d\tnbs=%d tpb=%d  counter_nodi0:%d num_nodi:%d MG_pesi:%d)\n",numAttivi,nbs, tpb, counter_nodi0, num_nodi, MG_pesi);fflush(stdout);
		kernel_EG_all_global_NEW1to2_8tpv<<<nbs, tpb>>>(0, counter_nodi0, numAttivi, MG_pesi);
		DEVSYNCANDCHECK_KER("kernel_EG_all_global (1) done")

		remove_nulls(hdev_nodeFlags2, num_nodi, &numAttivi);

		CUDASAFE( cudaMemset(hdev_nodeFlags1, 0, num_nodi*sizeof(hdev_nodeFlags1[0])) , "cudaMemset hdev_nodeFlags1");

		//printf("attivi=%d extloop=%d)\n",numAttivi,extloop);fflush(stdout);
		total_num_processed_nodes += (long)numAttivi;
		extloop--;
		if (numAttivi < 1) {break;}  // caso in cui la computazione termina in un numero dispari di fasi

		tpb = EVAL_TPB_PRAGMA(OTTO,NPRAG,numAttivi);
		nbs = MIN(MAX_BLOCKPERKERNEL, MYCEIL(OTTO*(MYCEIL(numAttivi,NPRAG)), tpb));
//		printf("attivi=%d\tnbs=%d tpb=%d  counter_nodi0:%d num_nodi:%d MG_pesi:%d)\n",numAttivi,nbs, tpb, counter_nodi0, num_nodi, MG_pesi);fflush(stdout);
		kernel_EG_all_global_NEW2to1_8tpv<<<nbs, tpb>>>(0, counter_nodi0, numAttivi, MG_pesi);
		DEVSYNCANDCHECK_KER("kernel_EG_all_global (2) done")

		remove_nulls(hdev_nodeFlags1, num_nodi, &numAttivi);

		CUDASAFE( cudaMemset(hdev_nodeFlags2, 0, num_nodi*sizeof(hdev_nodeFlags2[0])) , "cudaMemset hdev_nodeFlags2");

		//printf("attivi=%d extloop=%d)\n",numAttivi,extloop);fflush(stdout);
		total_num_processed_nodes += (long)numAttivi;
		extloop--;
//		printf("numAttivi: %d  residuo extloop=%ld\n",numAttivi,extloop);

		if (timeout_expired == 1) {break;}
	}
	printf("End EG on GPU after %ld loops (each loop involves one or more active nodes). Processed nodes %ld\n", max_loop-extloop, total_num_processed_nodes);
	statistics.processedNodes = total_num_processed_nodes;
//	cudaDeviceSynchronize();
}










/** \brief Calcolo su GPU shuffling 16 thread
 *
 * \details 
 *
 * \n Questo kernel si deve alternare con il kernel gemello che opera con i vettori dei flag scambiati
 * 
 **/
#define SEDICI 16

__global__ void kernel_EG_all_global_NEW1to2_16tpv(const int first, const int num_0nodes, int num_nodi_attivi, const int MG_pesi) {
	int tidx = THREAD_ID;
	int temp, val, idy, old, nodo;
	int aux1, aux2, aux3, aux4, aux5;
	uint off = (uint)(tidx%SEDICI);
	
	while (NPRAG*(tidx/SEDICI) <= num_nodi_attivi) { //var  num_nodi_attivi  e' nodi
	    #pragma unroll
	    for(int k = 0; k < NPRAG; k++) {
		if ((((tidx/SEDICI)*NPRAG) + k) >= num_nodi_attivi){break;}
		nodo = dev_nodeFlags1[(((tidx/SEDICI)*NPRAG) + k)+first] -SHIFTNOME;

		old = dev_ResNodeValues1[nodo];
		aux1 = dev_csrPtrInSuccLists[nodo];
		aux2 = dev_csrPtrInSuccLists[nodo+1];
		aux3 = dev_csrSuccLists[aux1];
		aux4 = dev_csrPesiArchi[aux1];
		aux5 = dev_ResNodeValues1[aux3];

		temp = OMINUS(aux5 , aux4);
		for (idy=aux1+1+off; idy < aux2; idy+=SEDICI) {  // una parte di lavoro a testa per i thread con off= 0 ... SEDICI-1 
			aux3 = dev_csrSuccLists[idy];
			aux4 = dev_csrPesiArchi[idy];
			aux5 = dev_ResNodeValues1[aux3];
			val = OMINUS(aux5 , aux4);
			if ((nodo<num_0nodes) && (temp > val)) { temp = val; }
			if ((nodo>=num_0nodes) && (temp < val)) { temp = val; }
		}

		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 8, SEDICI); 
		if (off<8) { // 0,...,7 aggiornano il proprio temp se il caso
	       		if ((nodo<num_0nodes) && (temp > aux5)) { temp = aux5; }
	       		if ((nodo>=num_0nodes) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 4, SEDICI); 
		if (off<4) { // 0,1,2,3 aggiornano il proprio temp se il caso
	       		if ((nodo<num_0nodes) && (temp > aux5)) { temp = aux5; }
	       		if ((nodo>=num_0nodes) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 2, SEDICI);  // i legge il temp  di i+2 (ha effetto per i=0,1)
		if (off<2) { // 0 e 1 aggiornano il proprio temp se il caso
	       		if ((nodo<num_0nodes) && (temp > aux5)) { temp = aux5; } 
	       		if ((nodo>=num_0nodes) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 1, SEDICI);  //0  legge il temp  di 1 (leggono anche gli altri ma e' ininfluente)
		if (off==0) { // 0 aggiorna il proprio temp se il caso
	       		if ((nodo<num_0nodes) && (temp > aux5)) { temp = aux5; }
	       		if ((nodo>=num_0nodes) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_sync(0xFFFFFFFF, temp, 0, SEDICI);  // tutti (0,...,7) leggono il temp  di 0

		//if (off==0) { // i due thread off=0 e off=1 sono nello stesso warp
	       		if ((nodo<num_0nodes) && (temp > aux5)) { temp = aux5; }
	       		if ((nodo>=num_0nodes) && (temp < aux5)) { temp = aux5; }
			if (old < temp) { // TODO:  li prendo tutti (da ottimizzare con "count")
				if (off==0) {dev_ResNodeValues1[nodo] = temp;}
				aux1 = dev_cscPtrInPredLists[nodo];
				aux2 = dev_cscPtrInPredLists[nodo+1];
				for (idy=aux1+off; idy < aux2; idy+=SEDICI) {
					aux3 = dev_cscPredLists[idy];
					//dev_nodeFlags2[aux3] = SHIFTNOME+aux3;  //RACE: qui diversi thread possono inserire lo stesso valore nella stessa posizione
					atomicCAS(dev_nodeFlags2+aux3, 0, SHIFTNOME+aux3); 
				}
			}
		//}
	    }
		tidx += ((blockDim.x * gridDim.x)*NPRAG);
	}
}


/** \brief Calcolo su GPU shuffling 16 thread
 *
 * \details 
 *
 * \n Questo kernel si deve alternare con il kernel gemello che opera con i vettori dei flag scambiati
 * 
 **/
__global__ void kernel_EG_all_global_NEW2to1_16tpv(const int first, const int num_0nodes, int num_nodi_attivi, const int MG_pesi) {
	int tidx = THREAD_ID;
	int temp, val, idy, old, nodo;
	int aux1, aux2, aux3, aux4, aux5;
	uint off = (uint)(tidx%SEDICI);
	
	while (NPRAG*(tidx/SEDICI) <= num_nodi_attivi) { //var  num_nodi_attivi  e' nodi
	    #pragma unroll
	    for(int k = 0; k < NPRAG; k++) {
		if ((((tidx/SEDICI)*NPRAG) + k) >= num_nodi_attivi){break;}
		nodo = dev_nodeFlags2[(((tidx/SEDICI)*NPRAG) + k)+first] -SHIFTNOME;

		old = dev_ResNodeValues1[nodo];
		aux1 = dev_csrPtrInSuccLists[nodo];
		aux2 = dev_csrPtrInSuccLists[nodo+1];
		aux3 = dev_csrSuccLists[aux1];
		aux4 = dev_csrPesiArchi[aux1];
		aux5 = dev_ResNodeValues1[aux3];

		temp = OMINUS(aux5 , aux4);
		for (idy=aux1+1+off; idy < aux2; idy+=SEDICI) {
			aux3 = dev_csrSuccLists[idy];
			aux4 = dev_csrPesiArchi[idy];
			aux5 = dev_ResNodeValues1[aux3];
			val = OMINUS(aux5 , aux4);
			if ((nodo<num_0nodes) && (temp > val)) { temp = val; }
			if ((nodo>=num_0nodes) && (temp < val)) { temp = val; }
		}
		
		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 8, SEDICI); 
		if (off<8) { // 0,...,7 aggiornano il proprio temp se il caso
	       		if ((nodo<num_0nodes) && (temp > aux5)) { temp = aux5; }
	       		if ((nodo>=num_0nodes) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 4, SEDICI); 
		if (off<4) { // 0,1,2,3 aggiornano il proprio temp se il caso
	       		if ((nodo<num_0nodes) && (temp > aux5)) { temp = aux5; }
	       		if ((nodo>=num_0nodes) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 2, SEDICI);  // i legge il temp  di i+2 (ha effetto per i=0,1)
		if (off<2) { // 0 e 1 aggiornano il proprio temp se il caso
	       		if ((nodo<num_0nodes) && (temp > aux5)) { temp = aux5; } 
	       		if ((nodo>=num_0nodes) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 1, SEDICI);  //0  legge il temp  di 1 (leggono anche gli altri ma e' ininfluente)
		if (off==0) { // 0 aggiorna il proprio temp se il caso
	       		if ((nodo<num_0nodes) && (temp > aux5)) { temp = aux5; }
	       		if ((nodo>=num_0nodes) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_sync(0xFFFFFFFF, temp, 0, SEDICI);  // tutti (0,...,7) leggono il temp  di 0

		//if (off==0) { // i due thread off=0 e off=1 sono nello stesso warp
	       		if ((nodo<num_0nodes) && (temp > aux5)) { temp = aux5; }
	       		if ((nodo>=num_0nodes) && (temp < aux5)) { temp = aux5; }
			if (old < temp) { // TODO:  li prendo tutti (da ottimizzare con "count")
				if (off==0) {dev_ResNodeValues1[nodo] = temp;}
				aux1 = dev_cscPtrInPredLists[nodo];
				aux2 = dev_cscPtrInPredLists[nodo+1];
				for (idy=aux1+off; idy < aux2; idy+=SEDICI) {
					aux3 = dev_cscPredLists[idy];
					//dev_nodeFlags1[aux3] = SHIFTNOME+aux3;  //RACE: qui diversi thread possono inserire lo stesso valore nella stessa posizione
					atomicCAS(dev_nodeFlags1+aux3, 0, SHIFTNOME+aux3); 
				}
			}
		//}
	    }
		tidx += ((blockDim.x * gridDim.x)*NPRAG);
	}
}





/** \brief Calcolo usando varie implementazioni di EG su GPU usando 16-shuffling
 *
 * \details ...
 *
 **/
void EG_gpu_solver_16() {

	long total_num_processed_nodes = 0;
	long max_loop = configuration.max_loop_val;
	long extloop;
	int numAttivi;

	int tpb = EVAL_TPB(1,num_nodi);
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


	printf("Running EG on GPU. (dev_EG_alg_shfl_full_16tpv.cu) (MG_pesi=%d max_loop=%ld extloop=%ld num nodes=%d max weight=%d tpb=%d)\n", MG_pesi, max_loop, extloop, num_nodi, max_pesi, tpb); fflush(stdout);

	numAttivi = num_nodi;
//	printf("attivi=%d\tnbs=%d tpb=%d  counter_nodi0:%d num_nodi:%d MG_pesi:%d)\n",numAttivi,nbs, tpb, counter_nodi0, num_nodi, MG_pesi);fflush(stdout);
	kernel_EG_initialize<<<nbs, tpb>>>(counter_nodi0, num_nodi, MG_pesi);
	DEVSYNCANDCHECK_KER("kernel_EG_initialize  done")

	remove_nulls(hdev_nodeFlags1, num_nodi, &numAttivi);
	//printf("attivi=%d extloop=%d)\n",numAttivi,extloop);fflush(stdout);
	total_num_processed_nodes += (long)numAttivi;

	while ((extloop>0) && (numAttivi>0)) {
		tpb = EVAL_TPB_PRAGMA(SEDICI,NPRAG,numAttivi);
		nbs = MIN(MAX_BLOCKPERKERNEL, MYCEIL(SEDICI*(MYCEIL(numAttivi,NPRAG)), tpb));

//		printf("attivi=%d\tnbs=%d tpb=%d  counter_nodi0:%d num_nodi:%d MG_pesi:%d)\n",numAttivi,nbs, tpb, counter_nodi0, num_nodi, MG_pesi);fflush(stdout);
		kernel_EG_all_global_NEW1to2_16tpv<<<nbs, tpb>>>(0, counter_nodi0, numAttivi, MG_pesi);
		DEVSYNCANDCHECK_KER("kernel_EG_all_global (1) done")

		remove_nulls(hdev_nodeFlags2, num_nodi, &numAttivi);

		CUDASAFE( cudaMemset(hdev_nodeFlags1, 0, num_nodi*sizeof(hdev_nodeFlags1[0])) , "cudaMemset hdev_nodeFlags1");

		//printf("attivi=%d extloop=%d)\n",numAttivi,extloop);fflush(stdout);
		total_num_processed_nodes += (long)numAttivi;
		extloop--;
		if (numAttivi < 1) {break;}  // caso in cui la computazione termina in un numero dispari di fasi

		tpb = EVAL_TPB_PRAGMA(SEDICI,NPRAG,numAttivi);
		nbs = MIN(MAX_BLOCKPERKERNEL, MYCEIL(SEDICI*(MYCEIL(numAttivi,NPRAG)), tpb));
//		printf("attivi=%d\tnbs=%d tpb=%d  counter_nodi0:%d num_nodi:%d MG_pesi:%d)\n",numAttivi,nbs, tpb, counter_nodi0, num_nodi, MG_pesi);fflush(stdout);
		kernel_EG_all_global_NEW2to1_16tpv<<<nbs, tpb>>>(0, counter_nodi0, numAttivi, MG_pesi);
		DEVSYNCANDCHECK_KER("kernel_EG_all_global (2) done")

		remove_nulls(hdev_nodeFlags1, num_nodi, &numAttivi);

		CUDASAFE( cudaMemset(hdev_nodeFlags2, 0, num_nodi*sizeof(hdev_nodeFlags2[0])) , "cudaMemset hdev_nodeFlags2");

		//printf("attivi=%d extloop=%d)\n",numAttivi,extloop);fflush(stdout);
		total_num_processed_nodes += (long)numAttivi;
		extloop--;
//		printf("numAttivi: %d  residuo extloop=%ld\n",numAttivi,extloop);

		if (timeout_expired == 1) {break;}
	}
	printf("End EG on GPU after %ld loops (each loop involves one or more active nodes). Processed nodes %ld\n", max_loop-extloop, total_num_processed_nodes);
	statistics.processedNodes = total_num_processed_nodes;
//	cudaDeviceSynchronize();
}







/** \brief Calcolo su GPU shuffling 32 thread
 *
 * \details 
 *
 * \n Questo kernel si deve alternare con il kernel gemello che opera con i vettori dei flag scambiati
 * 
 **/
#define TRENTADUE 32
__global__ void kernel_EG_all_global_NEW1to2_32tpv(const int first, const int num_0nodes, int num_nodi_attivi, const int MG_pesi) {
	int tidx = THREAD_ID;
	int temp, val, idy, old, nodo;
	int aux1, aux2, aux3, aux4, aux5;
	uint off = (uint)(tidx%TRENTADUE);
	
	while (NPRAG*(tidx/TRENTADUE) <= num_nodi_attivi) { //var  num_nodi_attivi  e' nodi
	    #pragma unroll
	    for(int k = 0; k < NPRAG; k++) {
		if ((((tidx/TRENTADUE)*NPRAG) + k) >= num_nodi_attivi){break;}
		nodo = dev_nodeFlags1[(((tidx/TRENTADUE)*NPRAG) + k)+first] -SHIFTNOME;

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

		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 16, TRENTADUE); 
		if (off<16) { // 0,...,15 aggiornano il proprio temp se il caso
	       		if ((nodo<num_0nodes) && (temp > aux5)) { temp = aux5; }
	       		if ((nodo>=num_0nodes) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 8, TRENTADUE); 
		if (off<8) { // 0,...,7 aggiornano il proprio temp se il caso
	       		if ((nodo<num_0nodes) && (temp > aux5)) { temp = aux5; }
	       		if ((nodo>=num_0nodes) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 4, TRENTADUE); 
		if (off<4) { // 0,1,2,3 aggiornano il proprio temp se il caso
	       		if ((nodo<num_0nodes) && (temp > aux5)) { temp = aux5; }
	       		if ((nodo>=num_0nodes) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 2, TRENTADUE);  // i legge il temp  di i+2 (ha effetto per i=0,1)
		if (off<2) { // 0 e 1 aggiornano il proprio temp se il caso
	       		if ((nodo<num_0nodes) && (temp > aux5)) { temp = aux5; } 
	       		if ((nodo>=num_0nodes) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 1, TRENTADUE);  //0  legge il temp  di 1 (leggono anche gli altri ma e' ininfluente)
		if (off==0) { // 0 aggiorna il proprio temp se il caso
	       		if ((nodo<num_0nodes) && (temp > aux5)) { temp = aux5; }
	       		if ((nodo>=num_0nodes) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_sync(0xFFFFFFFF, temp, 0, TRENTADUE);  // tutti (0,...,31) leggono il temp  di 0

		//if (off==0) { // i due thread off=0 e off=1 sono nello stesso warp
	       		if ((nodo<num_0nodes) && (temp > aux5)) { temp = aux5; }
	       		if ((nodo>=num_0nodes) && (temp < aux5)) { temp = aux5; }
			if (old < temp) { // TODO:  li prendo tutti (da ottimizzare con "count")
				if (off==0) {dev_ResNodeValues1[nodo] = temp;}
				aux1 = dev_cscPtrInPredLists[nodo];
				aux2 = dev_cscPtrInPredLists[nodo+1];
				for (idy=aux1+off; idy < aux2; idy+=TRENTADUE) {
					aux3 = dev_cscPredLists[idy];
					//dev_nodeFlags2[aux3] = SHIFTNOME+aux3;  //RACE: qui diversi thread possono inserire lo stesso valore nella stessa posizione
					atomicCAS(dev_nodeFlags2+aux3, 0, SHIFTNOME+aux3); 
				}
			}
		//}
	    }
	        tidx += ((blockDim.x * gridDim.x)*NPRAG);
		// NOPRAGMA:  tidx += (blockDim.x * gridDim.x);
	}
}


/** \brief Calcolo su GPU shuffling 32 thread
 *
 * \details 
 *
 * \n Questo kernel si deve alternare con il kernel gemello che opera con i vettori dei flag scambiati
 * 
 **/
__global__ void kernel_EG_all_global_NEW2to1_32tpv(const int first, const int num_0nodes, int num_nodi_attivi, const int MG_pesi) {
	int tidx = THREAD_ID;
	int temp, val, idy, old, nodo;
	int aux1, aux2, aux3, aux4, aux5;
	uint off = (uint)(tidx%TRENTADUE);
	
	while (NPRAG*(tidx/TRENTADUE) <= num_nodi_attivi) { //var  num_nodi_attivi  e' nodi
	    #pragma unroll
	    for(int k = 0; k < NPRAG; k++) {
		if ((((tidx/TRENTADUE)*NPRAG) + k) >= num_nodi_attivi){break;}
		nodo = dev_nodeFlags2[(((tidx/TRENTADUE)*NPRAG) + k)+first] -SHIFTNOME;
		// NOPRAGMA:  nodo = dev_nodeFlags2[tidx/TRENTADUE+first] -SHIFTNOME;

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
		
		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 16, TRENTADUE); 
		if (off<16) { // 0,...,15 aggiornano il proprio temp se il caso
	       		if ((nodo<num_0nodes) && (temp > aux5)) { temp = aux5; }
	       		if ((nodo>=num_0nodes) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 8, TRENTADUE); 
		if (off<8) { // 0,...,7 aggiornano il proprio temp se il caso
	       		if ((nodo<num_0nodes) && (temp > aux5)) { temp = aux5; }
	       		if ((nodo>=num_0nodes) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 4, TRENTADUE); 
		if (off<4) { // 0,1,2,3 aggiornano il proprio temp se il caso
	       		if ((nodo<num_0nodes) && (temp > aux5)) { temp = aux5; }
	       		if ((nodo>=num_0nodes) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 2, TRENTADUE);  // i legge il temp  di i+2 (ha effetto per i=0,1)
		if (off<2) { // 0 e 1 aggiornano il proprio temp se il caso
	       		if ((nodo<num_0nodes) && (temp > aux5)) { temp = aux5; } 
	       		if ((nodo>=num_0nodes) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 1, TRENTADUE);  //0  legge il temp  di 1 (leggono anche gli altri ma e' ininfluente)
		if (off==0) { // 0 aggiorna il proprio temp se il caso
	       		if ((nodo<num_0nodes) && (temp > aux5)) { temp = aux5; }
	       		if ((nodo>=num_0nodes) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_sync(0xFFFFFFFF, temp, 0, TRENTADUE);  // tutti leggono il temp  di 0

		//if (off==0) { // i due thread off=0 e off=1 sono nello stesso warp
	       		if ((nodo<num_0nodes) && (temp > aux5)) { temp = aux5; }
	       		if ((nodo>=num_0nodes) && (temp < aux5)) { temp = aux5; }
			if (old < temp) { // TODO:  li prendo tutti (da ottimizzare con "count")
				if (off==0) {dev_ResNodeValues1[nodo] = temp;}
				aux1 = dev_cscPtrInPredLists[nodo];
				aux2 = dev_cscPtrInPredLists[nodo+1];
				for (idy=aux1+off; idy < aux2; idy+=TRENTADUE) {
					aux3 = dev_cscPredLists[idy];
					//dev_nodeFlags1[aux3] = SHIFTNOME+aux3;  //RACE: qui diversi thread possono inserire lo stesso valore nella stessa posizione
					atomicCAS(dev_nodeFlags1+aux3, 0, SHIFTNOME+aux3); 
				}
			}
		//}
	    }
	        tidx += ((blockDim.x * gridDim.x)*NPRAG);
	}
}





/** \brief Calcolo usando varie implementazioni di EG su GPU usando 32-shuffling
 *
 * \details ...
 *
 **/
void EG_gpu_solver_32() {

	long total_num_processed_nodes = 0;
	long max_loop = configuration.max_loop_val;
	long extloop;
	int numAttivi;

	int tpb = EVAL_TPB(1,num_nodi);
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


	printf("Running EG on GPU. (dev_EG_alg_shfl_full_32tpv.cu) (MG_pesi=%d max_loop=%ld extloop=%ld num nodes=%d max weight=%d tpb=%d)\n", MG_pesi, max_loop, extloop, num_nodi, max_pesi, tpb); fflush(stdout);

	numAttivi = num_nodi;
//	printf("attivi=%d\tnbs=%d tpb=%d  counter_nodi0:%d num_nodi:%d MG_pesi:%d)\n",numAttivi,nbs, tpb, counter_nodi0, num_nodi, MG_pesi);fflush(stdout);
	kernel_EG_initialize<<<nbs, tpb>>>(counter_nodi0, num_nodi, MG_pesi);
	DEVSYNCANDCHECK_KER("kernel_EG_initialize  done")

	remove_nulls(hdev_nodeFlags1, num_nodi, &numAttivi);
	//printf("attivi=%d extloop=%d)\n",numAttivi,extloop);fflush(stdout);
	total_num_processed_nodes += (long)numAttivi;

	while ((extloop>0) && (numAttivi>0)) {
		tpb = EVAL_TPB_PRAGMA(TRENTADUE,NPRAG,numAttivi);
		nbs = MIN(MAX_BLOCKPERKERNEL, MYCEIL((TRENTADUE)*(MYCEIL(numAttivi,NPRAG)), tpb));

		//printf("attivi=%d\tnbs=%d tpb=%d  counter_nodi0:%d num_nodi:%d MG_pesi:%d)\n",numAttivi,nbs, tpb, counter_nodi0, num_nodi, MG_pesi);fflush(stdout);
		kernel_EG_all_global_NEW1to2_32tpv<<<nbs, tpb>>>(0, counter_nodi0, numAttivi, MG_pesi);
		DEVSYNCANDCHECK_KER("kernel_EG_all_global (1) done")

		remove_nulls(hdev_nodeFlags2, num_nodi, &numAttivi);

		CUDASAFE( cudaMemset(hdev_nodeFlags1, 0, num_nodi*sizeof(hdev_nodeFlags1[0])) , "cudaMemset hdev_nodeFlags1");

		//printf("attivi=%d extloop=%d)\n",numAttivi,extloop);fflush(stdout);
		total_num_processed_nodes += (long)numAttivi;
		extloop--;
		if (numAttivi < 1) {break;}  // caso in cui la computazione termina in un numero dispari di fasi

		tpb = EVAL_TPB_PRAGMA(TRENTADUE,NPRAG,numAttivi);
		nbs = MIN(MAX_BLOCKPERKERNEL, MYCEIL((TRENTADUE)*(MYCEIL(numAttivi,NPRAG)), tpb));
//		printf("attivi=%d\tnbs=%d tpb=%d  counter_nodi0:%d num_nodi:%d MG_pesi:%d)\n",numAttivi,nbs, tpb, counter_nodi0, num_nodi, MG_pesi);fflush(stdout);
		kernel_EG_all_global_NEW2to1_32tpv<<<nbs, tpb>>>(0, counter_nodi0, numAttivi, MG_pesi);
		DEVSYNCANDCHECK_KER("kernel_EG_all_global (2) done")

		remove_nulls(hdev_nodeFlags1, num_nodi, &numAttivi);

		CUDASAFE( cudaMemset(hdev_nodeFlags2, 0, num_nodi*sizeof(hdev_nodeFlags2[0])) , "cudaMemset hdev_nodeFlags2");

		//printf("attivi=%d extloop=%d)\n",numAttivi,extloop);fflush(stdout);
		total_num_processed_nodes += (long)numAttivi;
		extloop--;
//		printf("numAttivi: %d  residuo extloop=%ld\n",numAttivi,extloop);

		if (timeout_expired == 1) {break;}
	}
	printf("End EG on GPU after %ld loops (each loop involves one or more active nodes). Processed nodes %ld\n", max_loop-extloop, total_num_processed_nodes);
	statistics.processedNodes = total_num_processed_nodes;
//	cudaDeviceSynchronize();
}




/** \brief Calcolo usando varie implementazioni di EG su GPU usando vertex-par o 32-shuffle a seconda della percentuale di attivi
 *
 * \details ...
 *
 **/
void EG_gpu_solver_PercentageThreshold() {

	long total_num_processed_nodes = 0;
	long max_loop = configuration.max_loop_val;
	long extloop;
	int numAttivi;

	int tpb = EVAL_TPB(1,num_nodi);
	int nbs = MIN(MAX_BLOCKPERKERNEL, MYCEIL(num_nodi, tpb));




	max_loop = aggiorna_max_loop((long)num_archi, (long)num_nodi, (long)MG_pesi, max_loop);
	extloop=max_loop;


	printf("Transposing arena...\t");fflush(stdout);
	EG_gpu_traspose_graph();
	printf("...done\n");fflush(stdout);


	printf("Running EG on GPU. (no shuffling or 32-shuffing depending on threshold=%d%%) (MG_pesi=%d max_loop=%ld extloop=%ld num nodes=%d max weight=%d tpb=%d)\n", configuration.shuffleThreshold, MG_pesi, max_loop, extloop, num_nodi, max_pesi, tpb); fflush(stdout);


	numAttivi = num_nodi;
//	printf("attivi=%d\tnbs=%d tpb=%d  counter_nodi0:%d num_nodi:%d MG_pesi:%d)\n",numAttivi,nbs, tpb, counter_nodi0, num_nodi, MG_pesi);fflush(stdout);
	kernel_EG_initialize<<<nbs, tpb>>>(counter_nodi0, num_nodi, MG_pesi);
	DEVSYNCANDCHECK_KER("kernel_EG_initialize  done")

	remove_nulls(hdev_nodeFlags1, num_nodi, &numAttivi);
	//printf("attivi=%d extloop=%d)\n",numAttivi,extloop);fflush(stdout);
	total_num_processed_nodes += (long)numAttivi;

	while ((extloop>0) && (numAttivi>0)) {
		tpb = EVAL_TPB(TRENTADUE,numAttivi);
		nbs = MIN(MAX_BLOCKPERKERNEL, MYCEIL(TRENTADUE*numAttivi, tpb));

		if (((float) configuration.shuffleThreshold) < (100*((float)numAttivi)/((float)num_nodi))) {  // se percentuale di thread attivi elevata usa vertex-par
			kernel_EG_all_global_NEW1to2_none<<<nbs, tpb>>>(0, counter_nodi0, numAttivi, MG_pesi);
			DEVSYNCANDCHECK_KER("kernel_EG_all_global (no shuffle) (1) done")
		} else { // altrimenti shuffling con 32 thread per nodo
			kernel_EG_all_global_NEW1to2_32tpv<<<nbs, tpb>>>(0, counter_nodi0, TRENTADUE*numAttivi, MG_pesi);
				DEVSYNCANDCHECK_KER("kernel_EG_all_global (shuffle-32) (1) done")
		}

		remove_nulls(hdev_nodeFlags2, num_nodi, &numAttivi);

		CUDASAFE( cudaMemset(hdev_nodeFlags1, 0, num_nodi*sizeof(hdev_nodeFlags1[0])) , "cudaMemset hdev_nodeFlags1");

		//printf("attivi=%d extloop=%d)\n",numAttivi,extloop);fflush(stdout);
		total_num_processed_nodes += (long)numAttivi;
		extloop--;
		if (numAttivi < 1) {break;}  // caso in cui la computazione termina in un numero dispari di fasi

		tpb = EVAL_TPB(TRENTADUE,numAttivi);
		nbs = MIN(MAX_BLOCKPERKERNEL, MYCEIL(TRENTADUE*numAttivi, tpb));
		if (((float) configuration.shuffleThreshold) < (100*((float)numAttivi)/((float)num_nodi))) {  // se percentuale di thread attivi elevata usa vertex-par
			kernel_EG_all_global_NEW2to1_none<<<nbs, tpb>>>(0, counter_nodi0, numAttivi, MG_pesi);
			DEVSYNCANDCHECK_KER("kernel_EG_all_global (no shuffle) (2) done")
		} else {
			kernel_EG_all_global_NEW2to1_32tpv<<<nbs, tpb>>>(0, counter_nodi0, TRENTADUE*numAttivi, MG_pesi);
			DEVSYNCANDCHECK_KER("kernel_EG_all_global (shuffle-32) (2) done")
		}

		remove_nulls(hdev_nodeFlags1, num_nodi, &numAttivi);

		CUDASAFE( cudaMemset(hdev_nodeFlags2, 0, num_nodi*sizeof(hdev_nodeFlags2[0])) , "cudaMemset hdev_nodeFlags2");

		total_num_processed_nodes += (long)numAttivi;
		extloop--;

		if (timeout_expired == 1) {break;}
	}
	printf("End EG on GPU after %ld loops (each loop involves one or more active nodes). Processed nodes %ld\n", max_loop-extloop, total_num_processed_nodes);
	statistics.processedNodes = total_num_processed_nodes;
//	cudaDeviceSynchronize();
}









/* ************************************************************************** 
   KERNEL analoghi ai precedenti ma che utilizzano i vettori dev_nodeFlags[12] 
   predisposti con il partizionamento dettato da --outdegree
************************************************************************** */


__global__ void kernel_EG_all_global_NEW1to2_none_double(const int shufflesplit_index, const int num_0nodes_1, const int num_0nodes_2, int num_nodi_attivi1, int num_nodi_attivi, const int MG_pesi) {
	int tidx = THREAD_ID;
	int temp, val, idy, old, nodo;
	int aux1, aux2, aux3, aux4, aux5;
	
	while (tidx < num_nodi_attivi) {
		nodo = dev_nodeFlags1[(tidx<num_nodi_attivi1)?tidx:(tidx+shufflesplit_index-num_nodi_attivi1)] -SHIFTNOME;
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
			if (((nodo<num_0nodes_1) || ((nodo>=shufflesplit_index)&&(nodo<num_0nodes_2))) && (temp > val)) { temp = val; }
			if (((nodo>=num_0nodes_2) || ((nodo<shufflesplit_index)&&(nodo>=num_0nodes_1))) && (temp < val)) { temp = val; }
		}
		if (old < temp) { // TODO:  li prendo tutti (da ottimizzare con "count")
			dev_ResNodeValues1[nodo] = temp;
			aux1 = dev_cscPtrInPredLists[nodo];
			aux2 = dev_cscPtrInPredLists[nodo+1];
			for (idy=aux1; idy < aux2; idy++) {
				aux3 = dev_cscPredLists[idy];
				//dev_nodeFlags2[aux3] = SHIFTNOME+aux3;  //RACE: qui diversi thread possono inserire lo stesso valore nella stessa posizione
				atomicCAS(dev_nodeFlags2+aux3, 0, SHIFTNOME+aux3); 
			}
		}
		tidx += (blockDim.x * gridDim.x);
	}
}


__global__ void kernel_EG_all_global_NEW2to1_none_double(const int shufflesplit_index, const int num_0nodes_1, const int num_0nodes_2, int num_nodi_attivi1, int num_nodi_attivi, const int MG_pesi) {
	int tidx = THREAD_ID;
	int temp, val, idy, old, nodo;
	int aux1, aux2, aux3, aux4, aux5;
	
	while (tidx < num_nodi_attivi) {
		nodo = dev_nodeFlags2[(tidx<num_nodi_attivi1)?tidx:(tidx+shufflesplit_index-num_nodi_attivi1)] -SHIFTNOME;
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
			if (((nodo<num_0nodes_1) || ((nodo>=shufflesplit_index)&&(nodo<num_0nodes_2))) && (temp > val)) { temp = val; }
			if (((nodo>=num_0nodes_2) || ((nodo<shufflesplit_index)&&(nodo>=num_0nodes_1))) && (temp < val)) { temp = val; }
		}
		if (old < temp) { // TODO:  li prendo tutti (da ottimizzare con "count")
			dev_ResNodeValues1[nodo] = temp;
			aux1 = dev_cscPtrInPredLists[nodo];
			aux2 = dev_cscPtrInPredLists[nodo+1];
			for (idy=aux1; idy < aux2; idy++) {
				aux3 = dev_cscPredLists[idy];
				//dev_nodeFlags1[aux3] = SHIFTNOME+aux3;  //RACE: qui diversi thread possono inserire lo stesso valore nella stessa posizione
				atomicCAS(dev_nodeFlags1+aux3, 0, SHIFTNOME+aux3); 
			}
		}
		tidx += (blockDim.x * gridDim.x);
	}
}





__global__ void kernel_EG_all_global_NEW1to2_2tpv_double(const int shufflesplit_index, const int num_0nodes_1, const int num_0nodes_2, int num_nodi_attivi1, int num_nodi_attivi, const int MG_pesi) {
	int tidx = THREAD_ID;
	int temp, val, idy, old, nodo;
	int aux1, aux2, aux3, aux4, aux5;
	uint off = (uint)(tidx%DUE);
	
	while (tidx < num_nodi_attivi) {
		nodo = dev_nodeFlags1[((tidx/DUE)<num_nodi_attivi1)?(tidx/DUE):((tidx/DUE)+shufflesplit_index-num_nodi_attivi1)] -SHIFTNOME;
		old = dev_ResNodeValues1[nodo];
		aux1 = dev_csrPtrInSuccLists[nodo];
		aux2 = dev_csrPtrInSuccLists[nodo+1];
		aux3 = dev_csrSuccLists[aux1];
		aux4 = dev_csrPesiArchi[aux1];
		aux5 = dev_ResNodeValues1[aux3];

		temp = OMINUS(aux5 , aux4);
		for (idy=aux1+1+off; idy < aux2; idy+=DUE) {  // meta' lavoro a testa tra i due thread con off=0 e off=1 
			aux3 = dev_csrSuccLists[idy];
			aux4 = dev_csrPesiArchi[idy];
			aux5 = dev_ResNodeValues1[aux3];
			val = OMINUS(aux5 , aux4);
			if (((nodo<num_0nodes_1) || ((nodo>=shufflesplit_index)&&(nodo<num_0nodes_2))) && (temp > val)) { temp = val; }
			if (((nodo>=num_0nodes_2) || ((nodo<shufflesplit_index)&&(nodo>=num_0nodes_1))) && (temp < val)) { temp = val; }
		}
		
		// VECCHIO:  aux5 = __shfl_sync(0xFFFFFFFF, temp, (tidx%32)+1-2*off);  //1-off
                // RIMPIAZZATO DA:
		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 1, DUE);  //0  legge il temp  di 1 
		if (off==0) { // 0 aggiorna il proprio temp se il caso
			if (((nodo<num_0nodes_1) || ((nodo>=shufflesplit_index)&&(nodo<num_0nodes_2))) && (temp > aux5)) { temp = aux5; }
			if (((nodo>=num_0nodes_2) || ((nodo<shufflesplit_index)&&(nodo>=num_0nodes_1))) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_sync(0xFFFFFFFF, temp, 0, DUE);  // tutti (0,1) leggono il temp  di 0

		//if (off==0) { // i due thread off=0 e off=1 sono nello stesso warp
			if (((nodo<num_0nodes_1) || ((nodo>=shufflesplit_index)&&(nodo<num_0nodes_2))) && (temp > aux5)) { temp = aux5; }
			if (((nodo>=num_0nodes_2) || ((nodo<shufflesplit_index)&&(nodo>=num_0nodes_1))) && (temp < aux5)) { temp = aux5; }
			if (old < temp) { // TODO:  li prendo tutti (da ottimizzare con "count")
				if (off==0) {dev_ResNodeValues1[nodo] = temp;}
				aux1 = dev_cscPtrInPredLists[nodo];
				aux2 = dev_cscPtrInPredLists[nodo+1];
				for (idy=aux1+off; idy < aux2; idy+=DUE) {
					aux3 = dev_cscPredLists[idy];
					//dev_nodeFlags2[aux3] = SHIFTNOME+aux3;  //RACE: qui diversi thread possono inserire lo stesso valore nella stessa posizione
					atomicCAS(dev_nodeFlags2+aux3, 0, SHIFTNOME+aux3); 
				}
			}
		//}
		tidx += (blockDim.x * gridDim.x);
	}
}


__global__ void kernel_EG_all_global_NEW2to1_2tpv_double(const int shufflesplit_index, const int num_0nodes_1, const int num_0nodes_2, int num_nodi_attivi1, int num_nodi_attivi, const int MG_pesi) {
	int tidx = THREAD_ID;
	int temp, val, idy, old, nodo;
	int aux1, aux2, aux3, aux4, aux5;
	uint off = (uint)(tidx%DUE);
	
	while (tidx < num_nodi_attivi) {
		nodo = dev_nodeFlags2[((tidx/DUE)<num_nodi_attivi1)?(tidx/DUE):((tidx/DUE)+shufflesplit_index-num_nodi_attivi1)] -SHIFTNOME;
		old = dev_ResNodeValues1[nodo];
		aux1 = dev_csrPtrInSuccLists[nodo];
		aux2 = dev_csrPtrInSuccLists[nodo+1];
		aux3 = dev_csrSuccLists[aux1];
		aux4 = dev_csrPesiArchi[aux1];
		aux5 = dev_ResNodeValues1[aux3];

		temp = OMINUS(aux5 , aux4);
		for (idy=aux1+1+off; idy < aux2; idy+=DUE) {
			aux3 = dev_csrSuccLists[idy];
			aux4 = dev_csrPesiArchi[idy];
			aux5 = dev_ResNodeValues1[aux3];
			val = OMINUS(aux5 , aux4);
			if (((nodo<num_0nodes_1) || ((nodo>=shufflesplit_index)&&(nodo<num_0nodes_2))) && (temp > val)) { temp = val; }
			if (((nodo>=num_0nodes_2) || ((nodo<shufflesplit_index)&&(nodo>=num_0nodes_1))) && (temp < val)) { temp = val; }
		}
		
		// VECCHIO:  aux5 = __shfl_sync(0xFFFFFFFF, temp, (tidx%32)+1-2*off);  //1-off
                // RIMPIAZZATO DA:
		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 1, DUE);  //0  legge il temp  di 1 
		if (off==0) { // 0 aggiorna il proprio temp se il caso
			if (((nodo<num_0nodes_1) || ((nodo>=shufflesplit_index)&&(nodo<num_0nodes_2))) && (temp > aux5)) { temp = aux5; }
			if (((nodo>=num_0nodes_2) || ((nodo<shufflesplit_index)&&(nodo>=num_0nodes_1))) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_sync(0xFFFFFFFF, temp, 0, DUE);  // tutti (0,1) leggono il temp  di 0
		//if (off==0) { // i due thread off=0 e off=1 sono nello stesso warp
			if (((nodo<num_0nodes_1) || ((nodo>=shufflesplit_index)&&(nodo<num_0nodes_2))) && (temp > aux5)) { temp = aux5; }
			if (((nodo>=num_0nodes_2) || ((nodo<shufflesplit_index)&&(nodo>=num_0nodes_1))) && (temp < aux5)) { temp = aux5; }
			if (old < temp) { // TODO:  li prendo tutti (da ottimizzare con "count")
				if (off==0) {dev_ResNodeValues1[nodo] = temp;}
				aux1 = dev_cscPtrInPredLists[nodo];
				aux2 = dev_cscPtrInPredLists[nodo+1];
				for (idy=aux1+off; idy < aux2; idy+=DUE) {
					aux3 = dev_cscPredLists[idy];
					//dev_nodeFlags1[aux3] = SHIFTNOME+aux3;  //RACE: qui diversi thread possono inserire lo stesso valore nella stessa posizione
					atomicCAS(dev_nodeFlags1+aux3, 0, SHIFTNOME+aux3); 
				}
			}
		//}
		tidx += (blockDim.x * gridDim.x);
	}
}




__global__ void kernel_EG_all_global_NEW1to2_4tpv_double(const int shufflesplit_index, const int num_0nodes_1, const int num_0nodes_2, int num_nodi_attivi1, int num_nodi_attivi, const int MG_pesi) {
	int tidx = THREAD_ID;
	int temp, val, idy, old, nodo;
	int aux1, aux2, aux3, aux4, aux5;
	uint off = (uint)(tidx%QUATTRO);
	
	while (tidx < num_nodi_attivi) {
		nodo = dev_nodeFlags1[((tidx/QUATTRO)<num_nodi_attivi1)?(tidx/QUATTRO):((tidx/QUATTRO)+shufflesplit_index-num_nodi_attivi1)] -SHIFTNOME;
		old = dev_ResNodeValues1[nodo];
		aux1 = dev_csrPtrInSuccLists[nodo];
		aux2 = dev_csrPtrInSuccLists[nodo+1];
		aux3 = dev_csrSuccLists[aux1];
		aux4 = dev_csrPesiArchi[aux1];
		aux5 = dev_ResNodeValues1[aux3];

		temp = OMINUS(aux5 , aux4);
		for (idy=aux1+1+off; idy < aux2; idy+=QUATTRO) {  // una parte di lavoro a testa per i thread con off= 0 ... QUATTRO-1 
			aux3 = dev_csrSuccLists[idy];
			aux4 = dev_csrPesiArchi[idy];
			aux5 = dev_ResNodeValues1[aux3];
			val = OMINUS(aux5 , aux4);
			if (((nodo<num_0nodes_1) || ((nodo>=shufflesplit_index)&&(nodo<num_0nodes_2))) && (temp > val)) { temp = val; }
			if (((nodo>=num_0nodes_2) || ((nodo<shufflesplit_index)&&(nodo>=num_0nodes_1))) && (temp < val)) { temp = val; }
		}

		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 2, QUATTRO); // i legge il temp  di i+2 (ha effetto per i=0,1)
		if (off<2) { // 0 e 1 aggiornano il proprio temp se il caso
			if (((nodo<num_0nodes_1) || ((nodo>=shufflesplit_index)&&(nodo<num_0nodes_2))) && (temp > aux5)) { temp = aux5; }
			if (((nodo>=num_0nodes_2) || ((nodo<shufflesplit_index)&&(nodo>=num_0nodes_1))) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 1, QUATTRO);  //0  legge il temp  di 1 (leggono anche gli altri ma e' ininfluente)
		if (off==0) { // 0 aggiorna il proprio temp se il caso
			if (((nodo<num_0nodes_1) || ((nodo>=shufflesplit_index)&&(nodo<num_0nodes_2))) && (temp > aux5)) { temp = aux5; }
			if (((nodo>=num_0nodes_2) || ((nodo<shufflesplit_index)&&(nodo>=num_0nodes_1))) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_sync(0xFFFFFFFF, temp, 0, QUATTRO);  // tutti (0,1,2,3) leggono il temp  di 0

		//if (off==0) { // i due thread off=0 e off=1 sono nello stesso warp
			if (((nodo<num_0nodes_1) || ((nodo>=shufflesplit_index)&&(nodo<num_0nodes_2))) && (temp > aux5)) { temp = aux5; }
			if (((nodo>=num_0nodes_2) || ((nodo<shufflesplit_index)&&(nodo>=num_0nodes_1))) && (temp < aux5)) { temp = aux5; }
			if (old < temp) { // TODO:  li prendo tutti (da ottimizzare con "count")
				if (off==0) {dev_ResNodeValues1[nodo] = temp;}
				aux1 = dev_cscPtrInPredLists[nodo];
				aux2 = dev_cscPtrInPredLists[nodo+1];
				for (idy=aux1+off; idy < aux2; idy+=QUATTRO) {
					aux3 = dev_cscPredLists[idy];
					//dev_nodeFlags2[aux3] = SHIFTNOME+aux3;  //RACE: qui diversi thread possono inserire lo stesso valore nella stessa posizione
					atomicCAS(dev_nodeFlags2+aux3, 0, SHIFTNOME+aux3); 
				}
			}
		//}
		tidx += (blockDim.x * gridDim.x);
	}
}


__global__ void kernel_EG_all_global_NEW2to1_4tpv_double(const int shufflesplit_index, const int num_0nodes_1, const int num_0nodes_2, int num_nodi_attivi1, int num_nodi_attivi, const int MG_pesi) {
	int tidx = THREAD_ID;
	int temp, val, idy, old, nodo;
	int aux1, aux2, aux3, aux4, aux5;
	uint off = (uint)(tidx%QUATTRO);
	
	while (tidx < num_nodi_attivi) {
		nodo = dev_nodeFlags2[((tidx/QUATTRO)<num_nodi_attivi1)?(tidx/QUATTRO):((tidx/QUATTRO)+shufflesplit_index-num_nodi_attivi1)] -SHIFTNOME;
		old = dev_ResNodeValues1[nodo];
		aux1 = dev_csrPtrInSuccLists[nodo];
		aux2 = dev_csrPtrInSuccLists[nodo+1];
		aux3 = dev_csrSuccLists[aux1];
		aux4 = dev_csrPesiArchi[aux1];
		aux5 = dev_ResNodeValues1[aux3];

		temp = OMINUS(aux5 , aux4);
		for (idy=aux1+1+off; idy < aux2; idy+=QUATTRO) {
			aux3 = dev_csrSuccLists[idy];
			aux4 = dev_csrPesiArchi[idy];
			aux5 = dev_ResNodeValues1[aux3];
			val = OMINUS(aux5 , aux4);
			if (((nodo<num_0nodes_1) || ((nodo>=shufflesplit_index)&&(nodo<num_0nodes_2))) && (temp > val)) { temp = val; }
			if (((nodo>=num_0nodes_2) || ((nodo<shufflesplit_index)&&(nodo>=num_0nodes_1))) && (temp < val)) { temp = val; }
		}
		
		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 2, QUATTRO); // i legge il temp  di i+2 (ha effetto per i=0,1)
		if (off<2) { // 0 e 1 aggiornano il proprio temp se il caso
			if (((nodo<num_0nodes_1) || ((nodo>=shufflesplit_index)&&(nodo<num_0nodes_2))) && (temp > aux5)) { temp = aux5; }
			if (((nodo>=num_0nodes_2) || ((nodo<shufflesplit_index)&&(nodo>=num_0nodes_1))) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 1, QUATTRO);  //0  legge il temp  di 1 (leggono anche gli altri ma e' ininfluente)
		if (off==0) { // 0 aggiorna il proprio temp se il caso
			if (((nodo<num_0nodes_1) || ((nodo>=shufflesplit_index)&&(nodo<num_0nodes_2))) && (temp > aux5)) { temp = aux5; }
			if (((nodo>=num_0nodes_2) || ((nodo<shufflesplit_index)&&(nodo>=num_0nodes_1))) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_sync(0xFFFFFFFF, temp, 0, QUATTRO);  // tutti (0,1,2,3) leggono il temp  di 0

		//if (off==0) { // i due thread off=0 e off=1 sono nello stesso warp
			if (((nodo<num_0nodes_1) || ((nodo>=shufflesplit_index)&&(nodo<num_0nodes_2))) && (temp > aux5)) { temp = aux5; }
			if (((nodo>=num_0nodes_2) || ((nodo<shufflesplit_index)&&(nodo>=num_0nodes_1))) && (temp < aux5)) { temp = aux5; }
			if (old < temp) { // TODO:  li prendo tutti (da ottimizzare con "count")
				if (off==0) {dev_ResNodeValues1[nodo] = temp;}
				aux1 = dev_cscPtrInPredLists[nodo];
				aux2 = dev_cscPtrInPredLists[nodo+1];
				for (idy=aux1+off; idy < aux2; idy+=QUATTRO) {
					aux3 = dev_cscPredLists[idy];
					//dev_nodeFlags1[aux3] = SHIFTNOME+aux3;  //RACE: qui diversi thread possono inserire lo stesso valore nella stessa posizione
					atomicCAS(dev_nodeFlags1+aux3, 0, SHIFTNOME+aux3); 
				}
			}
		//}
		tidx += (blockDim.x * gridDim.x);
	}
}




__global__ void kernel_EG_all_global_NEW1to2_8tpv_double(const int shufflesplit_index, const int num_0nodes_1, const int num_0nodes_2, int num_nodi_attivi1, int num_nodi_attivi, const int MG_pesi) {
	int tidx = THREAD_ID;
	int temp, val, idy, old, nodo;
	int aux1, aux2, aux3, aux4, aux5;
	uint off = (uint)(tidx%OTTO);
	
	while (tidx < num_nodi_attivi) {
		nodo = dev_nodeFlags1[((tidx/OTTO)<num_nodi_attivi1)?(tidx/OTTO):((tidx/OTTO)+shufflesplit_index-num_nodi_attivi1)] -SHIFTNOME;
		old = dev_ResNodeValues1[nodo];
		aux1 = dev_csrPtrInSuccLists[nodo];
		aux2 = dev_csrPtrInSuccLists[nodo+1];
		aux3 = dev_csrSuccLists[aux1];
		aux4 = dev_csrPesiArchi[aux1];
		aux5 = dev_ResNodeValues1[aux3];

		temp = OMINUS(aux5 , aux4);
		for (idy=aux1+1+off; idy < aux2; idy+=OTTO) {  // una parte di lavoro a testa per i thread con off= 0 ... OTTO-1 
			aux3 = dev_csrSuccLists[idy];
			aux4 = dev_csrPesiArchi[idy];
			aux5 = dev_ResNodeValues1[aux3];
			val = OMINUS(aux5 , aux4);
			if (((nodo<num_0nodes_1) || ((nodo>=shufflesplit_index)&&(nodo<num_0nodes_2))) && (temp > val)) { temp = val; }
			if (((nodo>=num_0nodes_2) || ((nodo<shufflesplit_index)&&(nodo>=num_0nodes_1))) && (temp < val)) { temp = val; }
		}

		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 4, OTTO); 
		if (off<4) { // 0,1,2,3 aggiornano il proprio temp se il caso
			if (((nodo<num_0nodes_1) || ((nodo>=shufflesplit_index)&&(nodo<num_0nodes_2))) && (temp > aux5)) { temp = aux5; }
			if (((nodo>=num_0nodes_2) || ((nodo<shufflesplit_index)&&(nodo>=num_0nodes_1))) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 2, OTTO);  // i legge il temp  di i+2 (ha effetto per i=0,1)
		if (off<2) { // 0 e 1 aggiornano il proprio temp se il caso
			if (((nodo<num_0nodes_1) || ((nodo>=shufflesplit_index)&&(nodo<num_0nodes_2))) && (temp > aux5)) { temp = aux5; }
			if (((nodo>=num_0nodes_2) || ((nodo<shufflesplit_index)&&(nodo>=num_0nodes_1))) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 1, OTTO);  //0  legge il temp  di 1 (leggono anche gli altri ma e' ininfluente)
		if (off==0) { // 0 aggiorna il proprio temp se il caso
			if (((nodo<num_0nodes_1) || ((nodo>=shufflesplit_index)&&(nodo<num_0nodes_2))) && (temp > aux5)) { temp = aux5; }
			if (((nodo>=num_0nodes_2) || ((nodo<shufflesplit_index)&&(nodo>=num_0nodes_1))) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_sync(0xFFFFFFFF, temp, 0, OTTO);  // tutti (0,...,7) leggono il temp  di 0

		//if (off==0) { // i due thread off=0 e off=1 sono nello stesso warp
			if (((nodo<num_0nodes_1) || ((nodo>=shufflesplit_index)&&(nodo<num_0nodes_2))) && (temp > aux5)) { temp = aux5; }
			if (((nodo>=num_0nodes_2) || ((nodo<shufflesplit_index)&&(nodo>=num_0nodes_1))) && (temp < aux5)) { temp = aux5; }
			if (old < temp) { // TODO:  li prendo tutti (da ottimizzare con "count")
				if (off==0) {dev_ResNodeValues1[nodo] = temp;}
				aux1 = dev_cscPtrInPredLists[nodo];
				aux2 = dev_cscPtrInPredLists[nodo+1];
				for (idy=aux1+off; idy < aux2; idy+=OTTO) {
					aux3 = dev_cscPredLists[idy];
					//dev_nodeFlags2[aux3] = SHIFTNOME+aux3;  //RACE: qui diversi thread possono inserire lo stesso valore nella stessa posizione
					atomicCAS(dev_nodeFlags2+aux3, 0, SHIFTNOME+aux3); 
				}
			}
		//}
		tidx += (blockDim.x * gridDim.x);
	}
}


__global__ void kernel_EG_all_global_NEW2to1_8tpv_double(const int shufflesplit_index, const int num_0nodes_1, const int num_0nodes_2, int num_nodi_attivi1, int num_nodi_attivi, const int MG_pesi) {
	int tidx = THREAD_ID;
	int temp, val, idy, old, nodo;
	int aux1, aux2, aux3, aux4, aux5;
	uint off = (uint)(tidx%OTTO);
	
	while (tidx < num_nodi_attivi) {
		nodo = dev_nodeFlags2[((tidx/OTTO)<num_nodi_attivi1)?(tidx/OTTO):((tidx/OTTO)+shufflesplit_index-num_nodi_attivi1)] -SHIFTNOME;
		old = dev_ResNodeValues1[nodo];
		aux1 = dev_csrPtrInSuccLists[nodo];
		aux2 = dev_csrPtrInSuccLists[nodo+1];
		aux3 = dev_csrSuccLists[aux1];
		aux4 = dev_csrPesiArchi[aux1];
		aux5 = dev_ResNodeValues1[aux3];

		temp = OMINUS(aux5 , aux4);
		for (idy=aux1+1+off; idy < aux2; idy+=OTTO) {
			aux3 = dev_csrSuccLists[idy];
			aux4 = dev_csrPesiArchi[idy];
			aux5 = dev_ResNodeValues1[aux3];
			val = OMINUS(aux5 , aux4);
			if (((nodo<num_0nodes_1) || ((nodo>=shufflesplit_index)&&(nodo<num_0nodes_2))) && (temp > val)) { temp = val; }
			if (((nodo>=num_0nodes_2) || ((nodo<shufflesplit_index)&&(nodo>=num_0nodes_1))) && (temp < val)) { temp = val; }
		}
		
		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 4, OTTO); 
		if (off<4) { // 0,1,2,3 aggiornano il proprio temp se il caso
			if (((nodo<num_0nodes_1) || ((nodo>=shufflesplit_index)&&(nodo<num_0nodes_2))) && (temp > aux5)) { temp = aux5; }
			if (((nodo>=num_0nodes_2) || ((nodo<shufflesplit_index)&&(nodo>=num_0nodes_1))) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 2, OTTO);  // i legge il temp  di i+2 (ha effetto per i=0,1)
		if (off<2) { // 0 e 1 aggiornano il proprio temp se il caso
			if (((nodo<num_0nodes_1) || ((nodo>=shufflesplit_index)&&(nodo<num_0nodes_2))) && (temp > aux5)) { temp = aux5; }
			if (((nodo>=num_0nodes_2) || ((nodo<shufflesplit_index)&&(nodo>=num_0nodes_1))) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 1, OTTO);  //0  legge il temp  di 1 (leggono anche gli altri ma e' ininfluente)
		if (off==0) { // 0 aggiorna il proprio temp se il caso
			if (((nodo<num_0nodes_1) || ((nodo>=shufflesplit_index)&&(nodo<num_0nodes_2))) && (temp > aux5)) { temp = aux5; }
			if (((nodo>=num_0nodes_2) || ((nodo<shufflesplit_index)&&(nodo>=num_0nodes_1))) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_sync(0xFFFFFFFF, temp, 0, OTTO);  // tutti (0,...,7) leggono il temp  di 0

		//if (off==0) { // i due thread off=0 e off=1 sono nello stesso warp
			if (((nodo<num_0nodes_1) || ((nodo>=shufflesplit_index)&&(nodo<num_0nodes_2))) && (temp > aux5)) { temp = aux5; }
			if (((nodo>=num_0nodes_2) || ((nodo<shufflesplit_index)&&(nodo>=num_0nodes_1))) && (temp < aux5)) { temp = aux5; }
			if (old < temp) { // TODO:  li prendo tutti (da ottimizzare con "count")
				if (off==0) {dev_ResNodeValues1[nodo] = temp;}
				aux1 = dev_cscPtrInPredLists[nodo];
				aux2 = dev_cscPtrInPredLists[nodo+1];
				for (idy=aux1+off; idy < aux2; idy+=OTTO) {
					aux3 = dev_cscPredLists[idy];
					//dev_nodeFlags1[aux3] = SHIFTNOME+aux3;  //RACE: qui diversi thread possono inserire lo stesso valore nella stessa posizione
					atomicCAS(dev_nodeFlags1+aux3, 0, SHIFTNOME+aux3); 
				}
			}
		//}
		tidx += (blockDim.x * gridDim.x);
	}
}




__global__ void kernel_EG_all_global_NEW1to2_16tpv_double(const int shufflesplit_index, const int num_0nodes_1, const int num_0nodes_2, int num_nodi_attivi1, int num_nodi_attivi, const int MG_pesi) {
	int tidx = THREAD_ID;
	int temp, val, idy, old, nodo;
	int aux1, aux2, aux3, aux4, aux5;
	uint off = (uint)(tidx%SEDICI);
	
	while (tidx < num_nodi_attivi) {
		nodo = dev_nodeFlags1[((tidx/SEDICI)<num_nodi_attivi1)?(tidx/SEDICI):((tidx/SEDICI)+shufflesplit_index-num_nodi_attivi1)] -SHIFTNOME;
		old = dev_ResNodeValues1[nodo];
		aux1 = dev_csrPtrInSuccLists[nodo];
		aux2 = dev_csrPtrInSuccLists[nodo+1];
		aux3 = dev_csrSuccLists[aux1];
		aux4 = dev_csrPesiArchi[aux1];
		aux5 = dev_ResNodeValues1[aux3];

		temp = OMINUS(aux5 , aux4);
		for (idy=aux1+1+off; idy < aux2; idy+=SEDICI) {  // una parte di lavoro a testa per i thread con off= 0 ... SEDICI-1 
			aux3 = dev_csrSuccLists[idy];
			aux4 = dev_csrPesiArchi[idy];
			aux5 = dev_ResNodeValues1[aux3];
			val = OMINUS(aux5 , aux4);
			if (((nodo<num_0nodes_1) || ((nodo>=shufflesplit_index)&&(nodo<num_0nodes_2))) && (temp > val)) { temp = val; }
			if (((nodo>=num_0nodes_2) || ((nodo<shufflesplit_index)&&(nodo>=num_0nodes_1))) && (temp < val)) { temp = val; }
		}

		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 8, SEDICI); 
		if (off<8) { // 0,...,7 aggiornano il proprio temp se il caso
			if (((nodo<num_0nodes_1) || ((nodo>=shufflesplit_index)&&(nodo<num_0nodes_2))) && (temp > aux5)) { temp = aux5; }
			if (((nodo>=num_0nodes_2) || ((nodo<shufflesplit_index)&&(nodo>=num_0nodes_1))) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 4, SEDICI); 
		if (off<4) { // 0,1,2,3 aggiornano il proprio temp se il caso
			if (((nodo<num_0nodes_1) || ((nodo>=shufflesplit_index)&&(nodo<num_0nodes_2))) && (temp > aux5)) { temp = aux5; }
			if (((nodo>=num_0nodes_2) || ((nodo<shufflesplit_index)&&(nodo>=num_0nodes_1))) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 2, SEDICI);  // i legge il temp  di i+2 (ha effetto per i=0,1)
		if (off<2) { // 0 e 1 aggiornano il proprio temp se il caso
			if (((nodo<num_0nodes_1) || ((nodo>=shufflesplit_index)&&(nodo<num_0nodes_2))) && (temp > aux5)) { temp = aux5; }
			if (((nodo>=num_0nodes_2) || ((nodo<shufflesplit_index)&&(nodo>=num_0nodes_1))) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 1, SEDICI);  //0  legge il temp  di 1 (leggono anche gli altri ma e' ininfluente)
		if (off==0) { // 0 aggiorna il proprio temp se il caso
			if (((nodo<num_0nodes_1) || ((nodo>=shufflesplit_index)&&(nodo<num_0nodes_2))) && (temp > aux5)) { temp = aux5; }
			if (((nodo>=num_0nodes_2) || ((nodo<shufflesplit_index)&&(nodo>=num_0nodes_1))) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_sync(0xFFFFFFFF, temp, 0, SEDICI);  // tutti (0,...,7) leggono il temp  di 0

		//if (off==0) { // i due thread off=0 e off=1 sono nello stesso warp
			if (((nodo<num_0nodes_1) || ((nodo>=shufflesplit_index)&&(nodo<num_0nodes_2))) && (temp > aux5)) { temp = aux5; }
			if (((nodo>=num_0nodes_2) || ((nodo<shufflesplit_index)&&(nodo>=num_0nodes_1))) && (temp < aux5)) { temp = aux5; }
			if (old < temp) { // TODO:  li prendo tutti (da ottimizzare con "count")
				if (off==0) {dev_ResNodeValues1[nodo] = temp;}
				aux1 = dev_cscPtrInPredLists[nodo];
				aux2 = dev_cscPtrInPredLists[nodo+1];
				for (idy=aux1+off; idy < aux2; idy+=SEDICI) {
					aux3 = dev_cscPredLists[idy];
					//dev_nodeFlags2[aux3] = SHIFTNOME+aux3;  //RACE: qui diversi thread possono inserire lo stesso valore nella stessa posizione
					atomicCAS(dev_nodeFlags2+aux3, 0, SHIFTNOME+aux3); 
				}
			}
		//}
		tidx += (blockDim.x * gridDim.x);
	}
}


__global__ void kernel_EG_all_global_NEW2to1_16tpv_double(const int shufflesplit_index, const int num_0nodes_1, const int num_0nodes_2, int num_nodi_attivi1, int num_nodi_attivi, const int MG_pesi) {
	int tidx = THREAD_ID;
	int temp, val, idy, old, nodo;
	int aux1, aux2, aux3, aux4, aux5;
	uint off = (uint)(tidx%SEDICI);
	
	while (tidx < num_nodi_attivi) {
		nodo = dev_nodeFlags2[((tidx/SEDICI)<num_nodi_attivi1)?(tidx/SEDICI):((tidx/SEDICI)+shufflesplit_index-num_nodi_attivi1)] -SHIFTNOME;
		old = dev_ResNodeValues1[nodo];
		aux1 = dev_csrPtrInSuccLists[nodo];
		aux2 = dev_csrPtrInSuccLists[nodo+1];
		aux3 = dev_csrSuccLists[aux1];
		aux4 = dev_csrPesiArchi[aux1];
		aux5 = dev_ResNodeValues1[aux3];

		temp = OMINUS(aux5 , aux4);
		for (idy=aux1+1+off; idy < aux2; idy+=SEDICI) {
			aux3 = dev_csrSuccLists[idy];
			aux4 = dev_csrPesiArchi[idy];
			aux5 = dev_ResNodeValues1[aux3];
			val = OMINUS(aux5 , aux4);
			if (((nodo<num_0nodes_1) || ((nodo>=shufflesplit_index)&&(nodo<num_0nodes_2))) && (temp > val)) { temp = val; }
			if (((nodo>=num_0nodes_2) || ((nodo<shufflesplit_index)&&(nodo>=num_0nodes_1))) && (temp < val)) { temp = val; }
		}
		
		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 8, SEDICI); 
		if (off<8) { // 0,...,7 aggiornano il proprio temp se il caso
			if (((nodo<num_0nodes_1) || ((nodo>=shufflesplit_index)&&(nodo<num_0nodes_2))) && (temp > aux5)) { temp = aux5; }
			if (((nodo>=num_0nodes_2) || ((nodo<shufflesplit_index)&&(nodo>=num_0nodes_1))) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 4, SEDICI); 
		if (off<4) { // 0,1,2,3 aggiornano il proprio temp se il caso
			if (((nodo<num_0nodes_1) || ((nodo>=shufflesplit_index)&&(nodo<num_0nodes_2))) && (temp > aux5)) { temp = aux5; }
			if (((nodo>=num_0nodes_2) || ((nodo<shufflesplit_index)&&(nodo>=num_0nodes_1))) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 2, SEDICI);  // i legge il temp  di i+2 (ha effetto per i=0,1)
		if (off<2) { // 0 e 1 aggiornano il proprio temp se il caso
			if (((nodo<num_0nodes_1) || ((nodo>=shufflesplit_index)&&(nodo<num_0nodes_2))) && (temp > aux5)) { temp = aux5; }
			if (((nodo>=num_0nodes_2) || ((nodo<shufflesplit_index)&&(nodo>=num_0nodes_1))) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 1, SEDICI);  //0  legge il temp  di 1 (leggono anche gli altri ma e' ininfluente)
		if (off==0) { // 0 aggiorna il proprio temp se il caso
			if (((nodo<num_0nodes_1) || ((nodo>=shufflesplit_index)&&(nodo<num_0nodes_2))) && (temp > aux5)) { temp = aux5; }
			if (((nodo>=num_0nodes_2) || ((nodo<shufflesplit_index)&&(nodo>=num_0nodes_1))) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_sync(0xFFFFFFFF, temp, 0, SEDICI);  // tutti (0,...,7) leggono il temp  di 0

		//if (off==0) { // i due thread off=0 e off=1 sono nello stesso warp
			if (((nodo<num_0nodes_1) || ((nodo>=shufflesplit_index)&&(nodo<num_0nodes_2))) && (temp > aux5)) { temp = aux5; }
			if (((nodo>=num_0nodes_2) || ((nodo<shufflesplit_index)&&(nodo>=num_0nodes_1))) && (temp < aux5)) { temp = aux5; }
			if (old < temp) { // TODO:  li prendo tutti (da ottimizzare con "count")
				if (off==0) {dev_ResNodeValues1[nodo] = temp;}
				aux1 = dev_cscPtrInPredLists[nodo];
				aux2 = dev_cscPtrInPredLists[nodo+1];
				for (idy=aux1+off; idy < aux2; idy+=SEDICI) {
					aux3 = dev_cscPredLists[idy];
					//dev_nodeFlags1[aux3] = SHIFTNOME+aux3;  //RACE: qui diversi thread possono inserire lo stesso valore nella stessa posizione
					atomicCAS(dev_nodeFlags1+aux3, 0, SHIFTNOME+aux3); 
				}
			}
		//}
		tidx += (blockDim.x * gridDim.x);
	}
}



__global__ void kernel_EG_all_global_NEW1to2_32tpv_double(const int shufflesplit_index, const int num_0nodes_1, const int num_0nodes_2, int num_nodi_attivi1, int num_nodi_attivi, const int MG_pesi) {
	int tidx = THREAD_ID;
	int temp, val, idy, old, nodo;
	int aux1, aux2, aux3, aux4, aux5;
	uint off = (uint)(tidx%TRENTADUE);
	
	while (tidx < num_nodi_attivi) {
		nodo = dev_nodeFlags1[((tidx/TRENTADUE)<num_nodi_attivi1)?(tidx/TRENTADUE):((tidx/TRENTADUE)+shufflesplit_index-num_nodi_attivi1)] -SHIFTNOME;
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
			if (((nodo<num_0nodes_1) || ((nodo>=shufflesplit_index)&&(nodo<num_0nodes_2))) && (temp > val)) { temp = val; }
			if (((nodo>=num_0nodes_2) || ((nodo<shufflesplit_index)&&(nodo>=num_0nodes_1))) && (temp < val)) { temp = val; }
		}

		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 16, TRENTADUE); 
		if (off<16) { // 0,...,15 aggiornano il proprio temp se il caso
			if (((nodo<num_0nodes_1) || ((nodo>=shufflesplit_index)&&(nodo<num_0nodes_2))) && (temp > aux5)) { temp = aux5; }
			if (((nodo>=num_0nodes_2) || ((nodo<shufflesplit_index)&&(nodo>=num_0nodes_1))) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 8, TRENTADUE); 
		if (off<8) { // 0,...,7 aggiornano il proprio temp se il caso
			if (((nodo<num_0nodes_1) || ((nodo>=shufflesplit_index)&&(nodo<num_0nodes_2))) && (temp > aux5)) { temp = aux5; }
			if (((nodo>=num_0nodes_2) || ((nodo<shufflesplit_index)&&(nodo>=num_0nodes_1))) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 4, TRENTADUE); 
		if (off<4) { // 0,1,2,3 aggiornano il proprio temp se il caso
			if (((nodo<num_0nodes_1) || ((nodo>=shufflesplit_index)&&(nodo<num_0nodes_2))) && (temp > aux5)) { temp = aux5; }
			if (((nodo>=num_0nodes_2) || ((nodo<shufflesplit_index)&&(nodo>=num_0nodes_1))) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 2, TRENTADUE);  // i legge il temp  di i+2 (ha effetto per i=0,1)
		if (off<2) { // 0 e 1 aggiornano il proprio temp se il caso
			if (((nodo<num_0nodes_1) || ((nodo>=shufflesplit_index)&&(nodo<num_0nodes_2))) && (temp > aux5)) { temp = aux5; }
			if (((nodo>=num_0nodes_2) || ((nodo<shufflesplit_index)&&(nodo>=num_0nodes_1))) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 1, TRENTADUE);  //0  legge il temp  di 1 (leggono anche gli altri ma e' ininfluente)
		if (off==0) { // 0 aggiorna il proprio temp se il caso
			if (((nodo<num_0nodes_1) || ((nodo>=shufflesplit_index)&&(nodo<num_0nodes_2))) && (temp > aux5)) { temp = aux5; }
			if (((nodo>=num_0nodes_2) || ((nodo<shufflesplit_index)&&(nodo>=num_0nodes_1))) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_sync(0xFFFFFFFF, temp, 0, TRENTADUE);  // tutti (0,...,31) leggono il temp  di 0

		//if (off==0) { // i due thread off=0 e off=1 sono nello stesso warp
			if (((nodo<num_0nodes_1) || ((nodo>=shufflesplit_index)&&(nodo<num_0nodes_2))) && (temp > aux5)) { temp = aux5; }
			if (((nodo>=num_0nodes_2) || ((nodo<shufflesplit_index)&&(nodo>=num_0nodes_1))) && (temp < aux5)) { temp = aux5; }
			if (old < temp) { // TODO:  li prendo tutti (da ottimizzare con "count")
				if (off==0) {dev_ResNodeValues1[nodo] = temp;}
				aux1 = dev_cscPtrInPredLists[nodo];
				aux2 = dev_cscPtrInPredLists[nodo+1];
				for (idy=aux1+off; idy < aux2; idy+=TRENTADUE) {
					aux3 = dev_cscPredLists[idy];
					//dev_nodeFlags2[aux3] = SHIFTNOME+aux3;  //RACE: qui diversi thread possono inserire lo stesso valore nella stessa posizione
					atomicCAS(dev_nodeFlags2+aux3, 0, SHIFTNOME+aux3); 
				}
			}
		//}
		tidx += (blockDim.x * gridDim.x);
	}
}


__global__ void kernel_EG_all_global_NEW2to1_32tpv_double(const int shufflesplit_index, const int num_0nodes_1, const int num_0nodes_2, int num_nodi_attivi1, int num_nodi_attivi, const int MG_pesi) {
	int tidx = THREAD_ID;
	int temp, val, idy, old, nodo;
	int aux1, aux2, aux3, aux4, aux5;
	uint off = (uint)(tidx%TRENTADUE);
	
	while (tidx < num_nodi_attivi) {
		nodo = dev_nodeFlags2[((tidx/TRENTADUE)<num_nodi_attivi1)?(tidx/TRENTADUE):((tidx/TRENTADUE)+shufflesplit_index-num_nodi_attivi1)] -SHIFTNOME;
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
			if (((nodo<num_0nodes_1) || ((nodo>=shufflesplit_index)&&(nodo<num_0nodes_2))) && (temp > val)) { temp = val; }
			if (((nodo>=num_0nodes_2) || ((nodo<shufflesplit_index)&&(nodo>=num_0nodes_1))) && (temp < val)) { temp = val; }
		}
		
		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 16, TRENTADUE); 
		if (off<16) { // 0,...,15 aggiornano il proprio temp se il caso
			if (((nodo<num_0nodes_1) || ((nodo>=shufflesplit_index)&&(nodo<num_0nodes_2))) && (temp > aux5)) { temp = aux5; }
			if (((nodo>=num_0nodes_2) || ((nodo<shufflesplit_index)&&(nodo>=num_0nodes_1))) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 8, TRENTADUE); 
		if (off<8) { // 0,...,7 aggiornano il proprio temp se il caso
			if (((nodo<num_0nodes_1) || ((nodo>=shufflesplit_index)&&(nodo<num_0nodes_2))) && (temp > aux5)) { temp = aux5; }
			if (((nodo>=num_0nodes_2) || ((nodo<shufflesplit_index)&&(nodo>=num_0nodes_1))) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 4, TRENTADUE); 
		if (off<4) { // 0,1,2,3 aggiornano il proprio temp se il caso
			if (((nodo<num_0nodes_1) || ((nodo>=shufflesplit_index)&&(nodo<num_0nodes_2))) && (temp > aux5)) { temp = aux5; }
			if (((nodo>=num_0nodes_2) || ((nodo<shufflesplit_index)&&(nodo>=num_0nodes_1))) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 2, TRENTADUE);  // i legge il temp  di i+2 (ha effetto per i=0,1)
		if (off<2) { // 0 e 1 aggiornano il proprio temp se il caso
			if (((nodo<num_0nodes_1) || ((nodo>=shufflesplit_index)&&(nodo<num_0nodes_2))) && (temp > aux5)) { temp = aux5; }
			if (((nodo>=num_0nodes_2) || ((nodo<shufflesplit_index)&&(nodo>=num_0nodes_1))) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_down_sync(0xFFFFFFFF, temp, 1, TRENTADUE);  //0  legge il temp  di 1 (leggono anche gli altri ma e' ininfluente)
		if (off==0) { // 0 aggiorna il proprio temp se il caso
			if (((nodo<num_0nodes_1) || ((nodo>=shufflesplit_index)&&(nodo<num_0nodes_2))) && (temp > aux5)) { temp = aux5; }
			if (((nodo>=num_0nodes_2) || ((nodo<shufflesplit_index)&&(nodo>=num_0nodes_1))) && (temp < aux5)) { temp = aux5; }
		}
		aux5 = __shfl_sync(0xFFFFFFFF, temp, 0, TRENTADUE);  // tutti leggono il temp  di 0

		//if (off==0) { // i due thread off=0 e off=1 sono nello stesso warp
			if (((nodo<num_0nodes_1) || ((nodo>=shufflesplit_index)&&(nodo<num_0nodes_2))) && (temp > aux5)) { temp = aux5; }
			if (((nodo>=num_0nodes_2) || ((nodo<shufflesplit_index)&&(nodo>=num_0nodes_1))) && (temp < aux5)) { temp = aux5; }
			if (old < temp) { // TODO:  li prendo tutti (da ottimizzare con "count")
				if (off==0) {dev_ResNodeValues1[nodo] = temp;}
				aux1 = dev_cscPtrInPredLists[nodo];
				aux2 = dev_cscPtrInPredLists[nodo+1];
				for (idy=aux1+off; idy < aux2; idy+=TRENTADUE) {
					aux3 = dev_cscPredLists[idy];
					//dev_nodeFlags1[aux3] = SHIFTNOME+aux3;  //RACE: qui diversi thread possono inserire lo stesso valore nella stessa posizione
					atomicCAS(dev_nodeFlags1+aux3, 0, SHIFTNOME+aux3); 
				}
			}
		//}
		tidx += (blockDim.x * gridDim.x);
	}
}



/** \brief Calcolo usando varie implementazioni di EG su GPU usando due algoritmi (tra vertex-par o 2/4/8/16/32-shuffle) a seconda del outdegree dei nodi
 *
 * \details ...
 *
 **/
void EG_gpu_solver_OutdegreeSplit() {

	long total_num_processed_nodes = 0;
	long max_loop = configuration.max_loop_val;
	long extloop;
	int numAttivi;
	int numAttivi1;
	int numAttivi2;

	int tpb = EVAL_TPB(1,num_nodi);
	int nbs = MIN(MAX_BLOCKPERKERNEL, MYCEIL(num_nodi, tpb));


#ifdef MYDEBUG
	int randal;
	int * buffer = (int*)malloc((1+MAX(num_nodi,num_archi))*sizeof(int));
	printf("nbs=%d tpb=%d)\n",nbs, tpb);fflush(stdout);
	CUDASAFE( cudaMemset(hdev_nodeFlags1, 0, num_nodi*sizeof(hdev_nodeFlags1[0])) , "cudaMemset hdev_nodeFlags1");
	CUDASAFE( cudaMemset(hdev_nodeFlags2, 0, num_nodi*sizeof(hdev_nodeFlags2[0])) , "cudaMemset hdev_nodeFlags2");
	{int idx,idy; for (idx=0; idx<num_nodi; idx++) { printf("%d)\t%d\n", nomeExt_of_nomeInt[mapping[idx]], host_ResNodeValues1[idx]); 
						   for (idy=host_csrPtrInSuccLists[idx]; idy<host_csrPtrInSuccLists[idx+1]; idy++) {
							   printf("\t\t%d(%d)\n", nomeExt_of_nomeInt[mapping[host_csrSuccLists[idy]]],  host_csrPesiArchi[idy]);} }}
#endif



	max_loop = aggiorna_max_loop((long)num_archi, (long)num_nodi, (long)MG_pesi, max_loop);
	extloop=max_loop;


	printf("Transposing arena...\t");fflush(stdout);
	EG_gpu_traspose_graph();
	printf("...done\n");fflush(stdout);


	printf("Running EG on GPU. (no shuffling or K-shuffing depending on outdegree=%d (split_index:%d shfl_low:%d shfl_up:%d (split_thr:%d shfl_double:%d)) (MG_pesi=%d max_loop=%ld extloop=%ld num nodes=%d max weight=%d tpb=%d)\n", configuration.shuffleSplit_val, configuration.shuffleSplit_index, configuration.shuffleSplit_low, configuration.shuffleSplit_up, configuration.shuffleSplit_soglia, configuration.shuffleSplit_double, MG_pesi, max_loop, extloop, num_nodi, max_pesi, tpb); fflush(stdout);


	numAttivi1 = configuration.shuffleSplit_index;
	numAttivi2 = num_nodi-numAttivi1;
	numAttivi = numAttivi1+numAttivi2;
	kernel_EG_initialize_split<<<nbs, tpb>>>(configuration.shuffleSplit_index, counter_nodi0_1, counter_nodi0_2, num_nodi, MG_pesi);
	DEVSYNCANDCHECK_KER("kernel_EG_initialize_split  done")

	remove_nulls(hdev_nodeFlags1, configuration.shuffleSplit_index, &numAttivi1); //printf("attivi1=%d extloop=%ld)\n",numAttivi1,extloop);fflush(stdout);
	remove_nulls(hdev_nodeFlags1+configuration.shuffleSplit_index, num_nodi-configuration.shuffleSplit_index, &numAttivi2); //printf("attivi2=%d extloop=%ld)\n",numAttivi2,extloop);fflush(stdout);
	total_num_processed_nodes += (long)numAttivi1 + (long)numAttivi2;
	numAttivi = numAttivi1+numAttivi2;

#ifdef MYDEBUG
	{int idx,idy; for (idx=0; idx<num_nodi; idx++) { printf("%d)\t%d\n", nomeExt_of_nomeInt[mapping[idx]], host_ResNodeValues1[idx]); 
						   for (idy=host_csrPtrInSuccLists[idx]; idy<host_csrPtrInSuccLists[idx+1]; idy++) {
							   printf("\t\t%d(%d)\n", nomeExt_of_nomeInt[mapping[host_csrSuccLists[idy]]],  host_csrPesiArchi[idy]);} }}
PRINTDEBUGSTUFF(" DOPO INIT-NULL ")
#endif

	while ((extloop>0) && (numAttivi>0)) {
		if (numAttivi > configuration.shuffleSplit_soglia) { // se molti nodi split in due kernel (con diverso shuffling) a seconda del outdegree
			if (numAttivi1>0) {
				tpb = EVAL_TPB(configuration.shuffleSplit_low,numAttivi1);
				nbs = MIN(MAX_BLOCKPERKERNEL, MYCEIL(numAttivi1, tpb));
				switch (configuration.shuffleSplit_low){
					case 1:
						kernel_EG_all_global_NEW1to2_none<<<nbs, tpb>>>(0, counter_nodi0_1, numAttivi1, MG_pesi);
						DEVSYNCANDCHECK_KER("kernel_EG_all_global (no shuffle) (1) done")
						break;
					case 2:
						kernel_EG_all_global_NEW1to2_2tpv<<<nbs, tpb>>>(0, counter_nodi0_1, DUE*numAttivi1, MG_pesi);
						DEVSYNCANDCHECK_KER("kernel_EG_all_global (shuffle-2) (1) done")
						break;
					case 4:
						kernel_EG_all_global_NEW1to2_4tpv<<<nbs, tpb>>>(0, counter_nodi0_1, QUATTRO*numAttivi1, MG_pesi);
						DEVSYNCANDCHECK_KER("kernel_EG_all_global (shuffle-4) (1) done")
						break;
					case 8:
						kernel_EG_all_global_NEW1to2_8tpv<<<nbs, tpb>>>(0, counter_nodi0_1, OTTO*numAttivi1, MG_pesi);
						DEVSYNCANDCHECK_KER("kernel_EG_all_global (shuffle-8) (1) done")
						break;
					case 16:
						kernel_EG_all_global_NEW1to2_16tpv<<<nbs, tpb>>>(0, counter_nodi0_1, SEDICI*numAttivi1, MG_pesi);
						DEVSYNCANDCHECK_KER("kernel_EG_all_global (shuffle-16) (1) done")
						break;
					case 32:
						kernel_EG_all_global_NEW1to2_32tpv<<<nbs, tpb>>>(0, counter_nodi0_1, TRENTADUE*numAttivi1, MG_pesi);
						DEVSYNCANDCHECK_KER("kernel_EG_all_global (shuffle-32) (1) done")
						break;
				}
			}/*end if (numAttivi1>0)*/
			if (numAttivi2>0) {
				tpb = EVAL_TPB(configuration.shuffleSplit_up,numAttivi2);
				nbs = MIN(MAX_BLOCKPERKERNEL, MYCEIL(numAttivi2, tpb));
				switch (configuration.shuffleSplit_up){ 
					case 1:
						kernel_EG_all_global_NEW1to2_none<<<nbs, tpb>>>(configuration.shuffleSplit_index, configuration.shuffleSplit_index+counter_nodi0_2, numAttivi2, MG_pesi);
						DEVSYNCANDCHECK_KER("kernel_EG_all_global (no shuffle) (1b) done")
						break;
					case 2:
						kernel_EG_all_global_NEW1to2_2tpv<<<nbs, tpb>>>(configuration.shuffleSplit_index, configuration.shuffleSplit_index+counter_nodi0_2, DUE*numAttivi2, MG_pesi);
						DEVSYNCANDCHECK_KER("kernel_EG_all_global (shuffle-2) (1b) done")
						break;
					case 4:
						kernel_EG_all_global_NEW1to2_4tpv<<<nbs, tpb>>>(configuration.shuffleSplit_index, configuration.shuffleSplit_index+counter_nodi0_2, QUATTRO*numAttivi2, MG_pesi);
						DEVSYNCANDCHECK_KER("kernel_EG_all_global (shuffle-4) (1b) done")
						break;
					case 8:
						kernel_EG_all_global_NEW1to2_8tpv<<<nbs, tpb>>>(configuration.shuffleSplit_index, configuration.shuffleSplit_index+counter_nodi0_2, OTTO*numAttivi2, MG_pesi);
						DEVSYNCANDCHECK_KER("kernel_EG_all_global (shuffle-8) (1b) done")
						break;
					case 16:
//printf(" nbs:%d  %d   shuffleSplit_index:%d counter_nodi0_2:%d numAttivi2:%d\n", nbs, tpb, configuration.shuffleSplit_index, counter_nodi0_2, numAttivi2);
						kernel_EG_all_global_NEW1to2_16tpv<<<nbs, tpb>>>(configuration.shuffleSplit_index, configuration.shuffleSplit_index+counter_nodi0_2, SEDICI*numAttivi2, MG_pesi);
						DEVSYNCANDCHECK_KER("kernel_EG_all_global (shuffle-16) (1b) done")
						break;
					case 32:
						kernel_EG_all_global_NEW1to2_32tpv<<<nbs, tpb>>>(configuration.shuffleSplit_index, configuration.shuffleSplit_index+counter_nodi0_2, TRENTADUE*numAttivi2, MG_pesi);
						DEVSYNCANDCHECK_KER("kernel_EG_all_global (shuffle-32) (1b) done")
						break;
				}
			} /*end if (numAttivi2>0)*/

		} else { /* ELSE di  (numAttivi > configuration.shuffleSplit_soglia). Pochi nodi: uso un kernel solo (come da opzioni su linea di comando */
			tpb = EVAL_TPB(configuration.shuffleSplit_double,numAttivi);
			nbs = MIN(MAX_BLOCKPERKERNEL, MYCEIL(numAttivi, tpb));
			switch (configuration.shuffleSplit_double){
				case 1:
					kernel_EG_all_global_NEW1to2_none_double<<<nbs, tpb>>>(configuration.shuffleSplit_index, counter_nodi0_1, configuration.shuffleSplit_index+counter_nodi0_2, numAttivi1,numAttivi, MG_pesi);
					DEVSYNCANDCHECK_KER("kernel_EG_all_global (no shuffle) (d1) done")
					break;
				case 2:
					nbs = MIN(MAX_BLOCKPERKERNEL, MYCEIL(DUE*numAttivi, tpb));
					kernel_EG_all_global_NEW1to2_2tpv_double<<<nbs, tpb>>>(configuration.shuffleSplit_index, counter_nodi0_1, configuration.shuffleSplit_index+counter_nodi0_2, numAttivi1, DUE*numAttivi, MG_pesi);
					DEVSYNCANDCHECK_KER("kernel_EG_all_global (shuffle-2) (d1) done")
					break;
				case 4:
					nbs = MIN(MAX_BLOCKPERKERNEL, MYCEIL(QUATTRO*numAttivi, tpb));
					kernel_EG_all_global_NEW1to2_4tpv_double<<<nbs, tpb>>>(configuration.shuffleSplit_index, counter_nodi0_1, configuration.shuffleSplit_index+counter_nodi0_2, numAttivi1, QUATTRO*numAttivi, MG_pesi);
					DEVSYNCANDCHECK_KER("kernel_EG_all_global (shuffle-4) (d1) done")
					break;
				case 8:
					nbs = MIN(MAX_BLOCKPERKERNEL, MYCEIL(OTTO*numAttivi, tpb));
					kernel_EG_all_global_NEW1to2_8tpv_double<<<nbs, tpb>>>(configuration.shuffleSplit_index, counter_nodi0_1, configuration.shuffleSplit_index+counter_nodi0_2, numAttivi1, OTTO*numAttivi, MG_pesi);
					DEVSYNCANDCHECK_KER("kernel_EG_all_global (shuffle-8) (d1) done")
					break;
				case 16:
					nbs = MIN(MAX_BLOCKPERKERNEL, MYCEIL(SEDICI*numAttivi, tpb));
					kernel_EG_all_global_NEW1to2_16tpv_double<<<nbs, tpb>>>(configuration.shuffleSplit_index, counter_nodi0_1, configuration.shuffleSplit_index+counter_nodi0_2, numAttivi1, SEDICI*numAttivi, MG_pesi);
					DEVSYNCANDCHECK_KER("kernel_EG_all_global (shuffle-16) (d1) done")
					break;
				case 32:
					nbs = MIN(MAX_BLOCKPERKERNEL, MYCEIL(TRENTADUE*numAttivi, tpb));
					kernel_EG_all_global_NEW1to2_32tpv_double<<<nbs, tpb>>>(configuration.shuffleSplit_index, counter_nodi0_1, configuration.shuffleSplit_index+counter_nodi0_2, numAttivi1, TRENTADUE*numAttivi, MG_pesi);
					DEVSYNCANDCHECK_KER("kernel_EG_all_global (shuffle-32) (d1) done")
					break;
			}
		} /* ENDIF (numAttivi > configuration.shuffleSplit_soglia) */


		remove_nulls(hdev_nodeFlags2, configuration.shuffleSplit_index, &numAttivi1);
		remove_nulls(hdev_nodeFlags2+configuration.shuffleSplit_index, num_nodi-configuration.shuffleSplit_index, &numAttivi2);
		CUDASAFE( cudaMemset(hdev_nodeFlags1, 0, num_nodi*sizeof(hdev_nodeFlags1[0])) , "cudaMemset hdev_nodeFlags1");


		//printf("attivi=%d extloop=%ld)\n",numAttivi1+numAttivi2,extloop);fflush(stdout);
		total_num_processed_nodes += (long)numAttivi1 + (long)numAttivi2;
		numAttivi = numAttivi1+numAttivi2;
		extloop--;
		if (numAttivi < 1) {  // caso in cui la computazione termina in un numero dispari di fasi
			break;
		}

		if (numAttivi > configuration.shuffleSplit_soglia) { // se molti nodi split in due kernel (con diverso shuffling) a seconda del outdegree
			if (numAttivi1>0) {
				tpb = EVAL_TPB(configuration.shuffleSplit_low,numAttivi1);
				nbs = MIN(MAX_BLOCKPERKERNEL, MYCEIL(numAttivi1, tpb));
				switch (configuration.shuffleSplit_low){
					case 1:
						kernel_EG_all_global_NEW2to1_none<<<nbs, tpb>>>(0, counter_nodi0_1, numAttivi1, MG_pesi);
						DEVSYNCANDCHECK_KER("kernel_EG_all_global (no shuffle) (2) done")
						break;
					case 2:
						kernel_EG_all_global_NEW2to1_2tpv<<<nbs, tpb>>>(0, counter_nodi0_1, DUE*numAttivi1, MG_pesi);
						DEVSYNCANDCHECK_KER("kernel_EG_all_global (shuffle-2) (2) done")
						break;
					case 4:
						kernel_EG_all_global_NEW2to1_4tpv<<<nbs, tpb>>>(0, counter_nodi0_1, QUATTRO*numAttivi1, MG_pesi);
						DEVSYNCANDCHECK_KER("kernel_EG_all_global (shuffle-4) (2) done")
						break;
					case 8:
						kernel_EG_all_global_NEW2to1_8tpv<<<nbs, tpb>>>(0, counter_nodi0_1, OTTO*numAttivi1, MG_pesi);
						DEVSYNCANDCHECK_KER("kernel_EG_all_global (shuffle-8) (2) done")
						break;
					case 16:
						kernel_EG_all_global_NEW2to1_16tpv<<<nbs, tpb>>>(0, counter_nodi0_1, SEDICI*numAttivi1, MG_pesi);
						DEVSYNCANDCHECK_KER("kernel_EG_all_global (shuffle-16) (2) done")
						break;
					case 32:
						kernel_EG_all_global_NEW2to1_32tpv<<<nbs, tpb>>>(0, counter_nodi0_1, TRENTADUE*numAttivi1, MG_pesi);
						DEVSYNCANDCHECK_KER("kernel_EG_all_global (shuffle-32) (2) done")
					break;
				}
			} /*end if (numAttivi1>0)*/
			if (numAttivi2>0) {
				tpb = EVAL_TPB(configuration.shuffleSplit_up,numAttivi2);
				nbs = MIN(MAX_BLOCKPERKERNEL, MYCEIL(numAttivi2, tpb));
				switch (configuration.shuffleSplit_up){ 
					case 1:
						kernel_EG_all_global_NEW2to1_none<<<nbs, tpb>>>(configuration.shuffleSplit_index, configuration.shuffleSplit_index+counter_nodi0_2, numAttivi2, MG_pesi);
						DEVSYNCANDCHECK_KER("kernel_EG_all_global (no shuffle) (2) done")
						break;
					case 2:
						kernel_EG_all_global_NEW2to1_2tpv<<<nbs, tpb>>>(configuration.shuffleSplit_index, configuration.shuffleSplit_index+counter_nodi0_2, DUE*numAttivi2, MG_pesi);
						DEVSYNCANDCHECK_KER("kernel_EG_all_global (shuffle-2) (2) done")
						break;
					case 4:
						kernel_EG_all_global_NEW2to1_4tpv<<<nbs, tpb>>>(configuration.shuffleSplit_index, configuration.shuffleSplit_index+counter_nodi0_2, QUATTRO*numAttivi2, MG_pesi);
						DEVSYNCANDCHECK_KER("kernel_EG_all_global (shuffle-4) (2) done")
						break;
					case 8:
						kernel_EG_all_global_NEW2to1_8tpv<<<nbs, tpb>>>(configuration.shuffleSplit_index, configuration.shuffleSplit_index+counter_nodi0_2, OTTO*numAttivi2, MG_pesi);
						DEVSYNCANDCHECK_KER("kernel_EG_all_global (shuffle-8) (2) done")
						break;
					case 16:
						kernel_EG_all_global_NEW2to1_16tpv<<<nbs, tpb>>>(configuration.shuffleSplit_index, configuration.shuffleSplit_index+counter_nodi0_2, SEDICI*numAttivi2, MG_pesi);
						DEVSYNCANDCHECK_KER("kernel_EG_all_global (shuffle-16) (2) done")
						break;
					case 32:
						kernel_EG_all_global_NEW2to1_32tpv<<<nbs, tpb>>>(configuration.shuffleSplit_index, configuration.shuffleSplit_index+counter_nodi0_2, TRENTADUE*numAttivi2, MG_pesi);
						DEVSYNCANDCHECK_KER("kernel_EG_all_global (shuffle-32) (2) done")
						break;
				}
			} /*end if (numAttivi2>0)*/

		} else { /* ELSE di  (numAttivi > configuration.shuffleSplit_soglia). Pochi nodi: uso un kernel solo (come da opzioni su linea di comando */
			tpb = EVAL_TPB(configuration.shuffleSplit_double,numAttivi);
			nbs = MIN(MAX_BLOCKPERKERNEL, MYCEIL(numAttivi, tpb));
			switch (configuration.shuffleSplit_double){
				case 1:
					kernel_EG_all_global_NEW2to1_none_double<<<nbs, tpb>>>(configuration.shuffleSplit_index, counter_nodi0_1, configuration.shuffleSplit_index+counter_nodi0_2, numAttivi1,numAttivi, MG_pesi);
					DEVSYNCANDCHECK_KER("kernel_EG_all_global (no shuffle) (d2) done")
					break;
				case 2:
					kernel_EG_all_global_NEW2to1_2tpv_double<<<nbs, tpb>>>(configuration.shuffleSplit_index, counter_nodi0_1, configuration.shuffleSplit_index+counter_nodi0_2, numAttivi1, DUE*numAttivi, MG_pesi);
					DEVSYNCANDCHECK_KER("kernel_EG_all_global (shuffle-2) (d2) done")
					break;
				case 4:
					kernel_EG_all_global_NEW2to1_4tpv_double<<<nbs, tpb>>>(configuration.shuffleSplit_index, counter_nodi0_1, configuration.shuffleSplit_index+counter_nodi0_2, numAttivi1, QUATTRO*numAttivi, MG_pesi);
					DEVSYNCANDCHECK_KER("kernel_EG_all_global (shuffle-4) (d2) done")
					break;
				case 8:
					kernel_EG_all_global_NEW2to1_8tpv_double<<<nbs, tpb>>>(configuration.shuffleSplit_index, counter_nodi0_1, configuration.shuffleSplit_index+counter_nodi0_2, numAttivi1, OTTO*numAttivi, MG_pesi);
					DEVSYNCANDCHECK_KER("kernel_EG_all_global (shuffle-8) (d2) done")
					break;
				case 16:
					kernel_EG_all_global_NEW2to1_16tpv_double<<<nbs, tpb>>>(configuration.shuffleSplit_index, counter_nodi0_1, configuration.shuffleSplit_index+counter_nodi0_2, numAttivi1, SEDICI*numAttivi, MG_pesi);
					DEVSYNCANDCHECK_KER("kernel_EG_all_global (shuffle-16) (d2) done")
					break;
				case 32:
					kernel_EG_all_global_NEW2to1_32tpv_double<<<nbs, tpb>>>(configuration.shuffleSplit_index, counter_nodi0_1, configuration.shuffleSplit_index+counter_nodi0_2, numAttivi1, TRENTADUE*numAttivi, MG_pesi);
					DEVSYNCANDCHECK_KER("kernel_EG_all_global (shuffle-32) (d2) done")
					break;
			}
		} /* ENDIF (numAttivi > configuration.shuffleSplit_soglia) */


		remove_nulls(hdev_nodeFlags1, configuration.shuffleSplit_index, &numAttivi1);
		remove_nulls(hdev_nodeFlags1+configuration.shuffleSplit_index, num_nodi-configuration.shuffleSplit_index, &numAttivi2);
		CUDASAFE( cudaMemset(hdev_nodeFlags2, 0, num_nodi*sizeof(hdev_nodeFlags2[0])) , "cudaMemset hdev_nodeFlags2");


		total_num_processed_nodes += (long)numAttivi1 + (long)numAttivi2;
		numAttivi = numAttivi1+numAttivi2;
		extloop--;

		if (timeout_expired == 1) {break;}
	}
	printf("End EG on GPU after %ld loops (each loop involves one or more active nodes). Processed nodes %ld\n", max_loop-extloop, total_num_processed_nodes);
	statistics.processedNodes = total_num_processed_nodes;
//	cudaDeviceSynchronize();
}







#define FLAGDEV_EG_ALG_CU 1
#endif



