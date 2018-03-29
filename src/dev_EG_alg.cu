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
__global__ void kernel_EG_all_global_NEW1to2(const int num_0nodes, int num_nodi_attivi, const int MG_pesi) {
        int tidx = THREAD_ID;
        int temp, val, idy, old, nodo;
	int aux1, aux2, aux3, aux4, aux5;
	
        if (tidx < num_nodi_attivi) {
                nodo = dev_nodeFlags1[tidx] -SHIFTNOME;
		old = dev_ResNodeValues1[nodo];
		aux1 = dev_csrPtrInSuccLists[nodo];
		aux2 = dev_csrPtrInSuccLists[nodo+1];
                //solo se non fatto dal chiamante con  memset  dev_nodeFlags1[tidx] = 0;
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
				dev_nodeFlags2[aux3] = SHIFTNOME+aux3;
			}
		}
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
	
        if (tidx < num_nodi_attivi) {
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
				dev_nodeFlags1[aux3] = SHIFTNOME+aux3;
			}
		}
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

        if (tidx < num_nodi) { //un thread per ogni nodo
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
			dev_nodeFlags1[tidx]= (temp==1) ? (SHIFTNOME+tidx) : 0;
                } else {
			while ((temp==1) && (idy < dev_csrPtrInSuccLists[tidx+1])) {
				if (dev_csrPesiArchi[idy] < 0) {
					temp = 0;
				}
				idy++;
			}
			// set se almeno un outedge negativo altrimenti 0
			dev_nodeFlags1[tidx]= (temp==0) ? (SHIFTNOME+tidx) : 0;
                }
        }
}






/** \brief Calcolo su GPU dell'algoritmo EG node-based: IMPLEMENTA ALGORITMO SEMPLIFICATO (e quindi errato)
 *
 * \details Implementazione node-based.
 * \n Un thread per nodo. Ogni thread calcola il max (o min) dei valori relativi a tutti i suoi successori
 * con una scansione lineare.
 *
 * \n I thread di ogni blocco eseguono slice loops su thread-per-block nodi. 
 * I nodi sono quindi partizionati tra blocchi. 
 * Se il kernel viene eseguito da diversi blocchi lo scheduler determina in quali partizioni di nodi
 * e in che ordine far evolvere il calcolo. Non e' garantito che i valori di ResNodeValues[] evolvano alla
 * stessa "velocita'". 
 * 
 * \warning Il for-loop in un blocco viene interrotto quando sh_flag==0. Per effetto del partizionamento/scheduling descritto
 * sopra, potrebbe accadere che blocchi diversi eseguano un diverso numero di loop (inferiore o uguale a slice).
 * Quindi in totale per alcuni nodi potrebbe non effettuarsi il numero di loop richiesto. Conseguentemente
 * Porebbe essere necessatio compiera altri loop per raggiugere il punto fisso.
 * 
 **/
//CALL:kernel_EG_all_global_loop<<<nbs, tpb>>>(counter_nodi0, num_nodi, MG_pesi, configuration.loop_slice_for_EG);
__global__ void kernel_EG_all_global_loop(const int num_0nodes, const int num_nodi, const int MG_pesi, const long slice) {
	int tidx = THREAD_ID;
	int temp, val, idy, old;
	long loop;
	__shared__ int sh_flag;

	if (threadIdx.x == 0) { sh_flag=0; }
	__syncthreads();


	if (tidx < num_nodi) {
		for (loop=0; loop<slice; loop++) {
			old = dev_ResNodeValues1[tidx];
			temp = OMINUS(dev_ResNodeValues1[dev_csrSuccLists[dev_csrPtrInSuccLists[tidx]]] , dev_csrPesiArchi[dev_csrPtrInSuccLists[tidx]]);
			if (tidx<num_0nodes) {
				for (idy=(dev_csrPtrInSuccLists[tidx])+1; idy < dev_csrPtrInSuccLists[tidx+1]; idy++) {
					val = OMINUS(dev_ResNodeValues1[dev_csrSuccLists[idy]] , dev_csrPesiArchi[idy]);
					if (temp > val) {
						temp = val;
					}
				}
			} else {
				for (idy=(dev_csrPtrInSuccLists[tidx])+1; idy < dev_csrPtrInSuccLists[tidx+1]; idy++) {
					val = OMINUS(dev_ResNodeValues1[dev_csrSuccLists[idy]] , dev_csrPesiArchi[idy]);
					if (temp < val) {
						temp = val;
					}
				}
			}
			dev_ResNodeValues1[tidx] = temp;
			if (temp != old) {sh_flag=1;}
//			__syncthreads();
		}
	}
	__syncthreads();
	if (threadIdx.x == 0) { if (sh_flag==1) { (*dev_flag) = 1; } }
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



//	printf("Max xhared=%d   bytesPesiArchi=%d   bytesValues=%d   bytesSuccList=%d\n",configuration.sharedMemPerBlock,bytesPesiArchi,bytesValues,bytesSuccList);fflush(stdout);
	if (max_loop < 0) { // default (-1) indica un numero max pari al teorico 
		//CHECK OVERFLOW IN:  max_loop = ((long)num_archi)*((long)MG_pesi)+1;
		if (MG_pesi > (LONG_MAX-1)/num_archi) {
			// overflow handling
			sprintf(str,"%d * %d", num_archi, MG_pesi);
			exitWithError("Error too many loops: %s --> overflow\n", str);
		} else {
			max_loop = ((long)num_archi)*((long)MG_pesi)+1;  // MEMO: num_archi, non num_nodi ! ! !
		}
	}
	extloop=max_loop;


	printf("Transposing arena...\t");fflush(stdout);
	EG_gpu_traspose_graph();
	printf("...done\n");fflush(stdout);


	if (configuration.loop_slice_for_EG != 1) {printf("WARNING: configuration.loop_slice_for_EG=%d  ignored!\n",configuration.loop_slice_for_EG);fflush(stdout);}
	if (configuration.loop_slice_for_EG < -1 ) {  //" > 1)"  DISABILITATO: implementazione kernel_EG_all_global_loop errata
		(*host_flag)=1;
		printf("Running EG_loop on GPU. (MG_pesi=%d max_loop=%ld extloop=%ld slice=%d num nodes=%d max weight=%d nbs=%d tpb=%d)\n", MG_pesi, max_loop, extloop, configuration.loop_slice_for_EG, num_nodi, max_pesi, nbs, tpb); fflush(stdout);
		while ((extloop>=configuration.loop_slice_for_EG) && ((*host_flag)!=0)) {
	//		printf("\n(prea) after %ld calls (V2):  max_loop=%ld extloop=%ld slice=%d flag=%d\n",(max_loop-extloop), max_loop, extloop, configuration.loop_slice_for_EG, (*host_flag));fflush(stdout);
			cudaMemset(hdev_flag, 0, sizeof(int));
			kernel_EG_all_global_loop<<<nbs, tpb>>>(counter_nodi0, num_nodi, MG_pesi, configuration.loop_slice_for_EG);
			DEVSYNCANDCHECK_KER("kernel_EG_all_global_loop (1) done")
			extloop -= configuration.loop_slice_for_EG;
		// per compensare il disallineamento dei blocks, basare il while solo su flag e attivare la riga seguente
		//	kernel_EG_all_global<<<nbs, tpb>>>(counter_nodi0, num_nodi, MG_pesi);
		//	DEVSYNCANDCHECK_KER("kernel_EG_all_global (X) done")
			cudaMemcpy(host_flag, hdev_flag, sizeof(int),cudaMemcpyDeviceToHost);
		}
		if ((extloop>0) && ((*host_flag)!=0)) {
			//printf("\n(b) after %ld calls (V2) last: \n",(max_loop-extloop));fflush(stdout);
//			printf("\n(b) after %ld calls (V2):  max_loop=%ld extloop=%ld slice=%d flag=%d\n",(max_loop-extloop), max_loop, extloop, configuration.loop_slice_for_EG, (*host_flag));fflush(stdout);
			kernel_EG_all_global_loop<<<1, configuration.threadsPerBlock>>>(counter_nodi0, num_nodi, MG_pesi, extloop);
			DEVSYNCANDCHECK_KER("kernel_EG_all_global_loop (2) done")
		}
	} else {
		printf("Running EG on GPU. (dev_EG_alg_shfl_none.cu) (MG_pesi=%d max_loop=%ld extloop=%ld slice=%d num nodes=%d max weight=%d tpb=%d)\n", MG_pesi, max_loop, extloop, configuration.loop_slice_for_EG, num_nodi, max_pesi, tpb); fflush(stdout);

		numAttivi = num_nodi;
//		printf("attivi=%d\tnbs=%d tpb=%d  counter_nodi0:%d num_nodi:%d MG_pesi:%d)\n",numAttivi,nbs, tpb, counter_nodi0, num_nodi, MG_pesi);fflush(stdout);
		kernel_EG_initialize<<<nbs, tpb>>>(counter_nodi0, num_nodi, MG_pesi);
		DEVSYNCANDCHECK_KER("kernel_EG_initialize  done")
		PRINTDEBUGSTUFF("dopo kernel_EG_initialize")

		remove_nulls(hdev_nodeFlags1, num_nodi, &numAttivi);
		PRINTDEBUGSTUFF("dopo remove_nulls iniziale")

		while ((extloop>0) && (numAttivi>0)) {
			//kernel_EG_all_global_OLD<<<nbs, tpb>>>(counter_nodi0, num_nodi, MG_pesi);
			//kernel_EG_all_global<<<nbs, tpb>>>(counter_nodi0, num_nodi, MG_pesi);
			tpb = (numAttivi< (int)configuration.threadsPerBlock) ? MIN(configuration.threadsPerBlock , MAX(configuration.warpSize, MYCEILSTEP(numAttivi, configuration.warpSize))) : configuration.threadsPerBlock;
			nbs = MIN(MAX_BLOCKPERKERNEL, MYCEIL(numAttivi, tpb));

//			printf("attivi=%d\tnbs=%d tpb=%d  counter_nodi0:%d num_nodi:%d MG_pesi:%d)\n",numAttivi,nbs, tpb, counter_nodi0, num_nodi, MG_pesi);fflush(stdout);
			kernel_EG_all_global_NEW1to2<<<nbs, tpb>>>(counter_nodi0, numAttivi, MG_pesi);
			DEVSYNCANDCHECK_KER("kernel_EG_all_global (1) done")
			PRINTDEBUGSTUFF("dopo kernel_EG_all_global_NEW1to2")

			remove_nulls(hdev_nodeFlags2, num_nodi, &numAttivi);
			PRINTDEBUGSTUFF("dopo remove_nulls nodeFlags2")

			CUDASAFE( cudaMemset(hdev_nodeFlags1, 0, num_nodi*sizeof(hdev_nodeFlags1[0])) , "cudaMemset hdev_nodeFlags1");
			PRINTDEBUGSTUFF("dopo cudaMemset  nodeFlags1")

			extloop--;
			if (numAttivi < 1) {break;}  // caso in cui la computazione termina in un numero dispari di fasi

			tpb = (numAttivi< (int)configuration.threadsPerBlock) ? MIN(configuration.threadsPerBlock , MAX(configuration.warpSize, MYCEILSTEP(numAttivi, configuration.warpSize))) : configuration.threadsPerBlock;
			nbs = MIN(MAX_BLOCKPERKERNEL, MYCEIL(numAttivi, tpb));
//			printf("attivi=%d\tnbs=%d tpb=%d  counter_nodi0:%d num_nodi:%d MG_pesi:%d)\n",numAttivi,nbs, tpb, counter_nodi0, num_nodi, MG_pesi);fflush(stdout);
			kernel_EG_all_global_NEW2to1<<<nbs, tpb>>>(counter_nodi0, numAttivi, MG_pesi);
			DEVSYNCANDCHECK_KER("kernel_EG_all_global (2) done")
			PRINTDEBUGSTUFF("dopo kernel_EG_all_global_NEW2to1")

			remove_nulls(hdev_nodeFlags1, num_nodi, &numAttivi);
			PRINTDEBUGSTUFF("dopo remove_nulls nodeFlags1")

			CUDASAFE( cudaMemset(hdev_nodeFlags2, 0, num_nodi*sizeof(hdev_nodeFlags2[0])) , "cudaMemset hdev_nodeFlags2");
			PRINTDEBUGSTUFF("dopo cudaMemset  nodeFlags2")

			extloop--;
//			printf("numAttivi: %d  residuo extloop=%ld\n",numAttivi,extloop);

			if (timeout_expired == 1) {break;}
		}
	}
//	cudaDeviceSynchronize();
}






#define FLAGDEV_EG_ALG_CU 1
#endif

