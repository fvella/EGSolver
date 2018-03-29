/** \file  dev_solver.cu
 * \brief Wrapper per le diverse implementazioni di solver paralleli 
 *
 */

#ifndef FLAGDEV_SOLVER_CU



//#include <cuda.h>

//#include "common.h"
#include "dev_common.h"
#include "dev_solver.h"
#include "utils.cu"

extern char* str;
extern int num_nodi;
extern int max_pesi;
extern int MG_pesi;
extern int num_archi;
extern int counter_nodi0;
extern int max_outdegree;
extern int *host_allData;
extern int *host_csrPtrInSuccLists;
extern int *host_csrSuccLists;
extern int *host_revSuccLists;
extern int *host_csrPesiArchi;
extern int *host_ResNodeValues1;
extern int *host_ResNodeValues2;
extern int *host_ResNodeValuesAux;
extern int *host_flag;
extern int *hdev_allData;
extern int *hdev_csrPtrInSuccLists;
extern int *hdev_csrSuccLists;
extern int *hdev_revSuccLists;
extern int *hdev_csrPesiArchi;
extern int *hdev_ResNodeValues1;
extern int *hdev_ResNodeValues2;
extern int *hdev_ResNodeValuesAux;
extern int *hdev_flag;
extern uint timeout_expired;
extern int *hdev_nodeFlags1;
extern int *hdev_nodeFlags2;

extern int *host_csrDataArchiAux;
extern int *hdev_csrDataArchiAux;

extern int *csrPtrInSuccLists;
extern int *nodePriority;
extern char *nodeOwner;
extern char *nodeFlags;
extern int *csrPesiArchi;
extern int *csrSuccLists;
extern int *revSuccLists;
extern int *nomeExt_of_nomeInt;
extern int *nomeInt_of_nomeExt;
extern int *mapping;
extern int *revmapping;
extern char **nodeName;

extern int *host_transData;
extern int *host_cscPtrInPredLists;
extern int *host_cscPredLists;
extern int *host_cscPesiArchiPred;

extern int *hdev_transData;
extern int *hdev_cscPtrInPredLists;
extern int *hdev_cscPredLists;
extern int *hdev_cscPesiArchiPred;

extern config configuration;
extern stat statistics;


#include "dev_EG_alg.cu"


void gpu_solver() {
	cudaEvent_t cuSolveStart, cuSolveStop;
	float solvetime;
	cudaEventCreate(&cuSolveStart);
	cudaEventCreate(&cuSolveStop);

	cudaEventRecord(cuSolveStart, 0);

	switch (configuration.algoritmo) {
		case ALGOR_EG0:  // wrap sulla versione --eg su gpu
			EG_gpu_solver();
			break;
		case ALGOR_EG:
			EG_gpu_solver();
			break;
		default:
			EG_gpu_solver();
			break;
	}

	cudaEventRecord(cuSolveStop, 0);
	cudaEventSynchronize(cuSolveStop);
	cudaEventElapsedTime(&solvetime, cuSolveStart, cuSolveStop);
	statistics.solvingtime = solvetime;
	
	//recupera risultati da device:
	cudaDeviceSynchronize();
	CUDASAFE( cudaMemcpy(host_ResNodeValues1, hdev_ResNodeValues1, num_nodi*sizeof(int),cudaMemcpyDeviceToHost) , "cudaMemcpyDeviceToHost host_ResNodeValues1");
	//cudaDeviceSynchronize();

}


int copia_dati_su_device() {
        int num_total_mem = num_nodi+1+num_archi+num_archi+num_nodi+num_nodi +1 +1; //(1 per i flag) 
//	printf("\tCOPIA DATI num_nodi=%d\n",num_nodi);fflush(stdout);
	CUDASAFE( cudaMemcpy(hdev_allData, host_allData, num_total_mem*sizeof(int),cudaMemcpyHostToDevice) , "cudaMemcpyHostToDevice dev_allData");

//	printf("\tCOPIATI\n");fflush(stdout);
	return(0);
}


int alloca_memoria_host() {
        int num_total_mem = num_nodi+1+num_archi+num_archi+num_nodi+num_nodi +1 +1; //(1 per i flag)

	num_total_mem += num_archi; // per csrDataArchiAux
	CUDASAFE( cudaMallocHost((void**)&host_allData, num_total_mem*sizeof(int)) , "cudaMallocHost: host_allData[]");
	host_csrPtrInSuccLists = host_allData;
	host_csrSuccLists = host_allData+num_nodi+1;
	host_csrPesiArchi = host_csrSuccLists+num_archi;
	host_ResNodeValues1 = host_csrPesiArchi+num_archi; 
	if ((configuration.algoritmo == ALGOR_EG0)) {
		host_ResNodeValues2 = host_ResNodeValues1+num_nodi; }
	else { host_ResNodeValues2 = host_ResNodeValues1; }
	host_flag = host_ResNodeValues2+num_nodi; 
	host_csrDataArchiAux = host_flag+1; 

	memset(host_ResNodeValues1, 0, num_nodi*sizeof(int)); //azzera vettore risultati

	if ((configuration.algoritmo == ALGOR_EG) || (configuration.algoritmo == ALGOR_EG0)) {
		host_ResNodeValuesAux = NULL;
		// NON USATO host_ResNodeValuesAux = host_csrDataArchiAux + num_archi;   // spazio addizionale per ResNodeValuesAux[]
	} else {
		host_revSuccLists = NULL;
		host_ResNodeValuesAux = NULL;
	}

	if ((configuration.algoritmo == ALGOR_EG) || (configuration.algoritmo == ALGOR_EG0)) {
		int num_trans_mem = num_nodi+1+(2*num_archi);
		//CUDASAFE( cudaMalloc((void **)&host_transData, num_trans_mem*sizeof(int)) , "cudaMalloc: &host_transData[]");
		host_transData = (int *)malloc(num_trans_mem*sizeof(int));  checkNullAllocation(host_transData,"allocazione host_transData");
		host_cscPtrInPredLists = host_transData;
		host_cscPredLists = host_transData + num_nodi+1;
		host_cscPesiArchiPred = host_cscPredLists + num_archi;
	}

	return(0);
}





int alloca_memoria_device() {
	int num_total_mem = num_nodi+1+num_archi+num_archi+num_nodi+num_nodi +1 +1; //(1 per i flag)

	CUDASAFE( cudaMalloc((void **)&hdev_allData, num_total_mem*sizeof(int)) , "cudaMalloc: &hdev_allData[]");
	hdev_csrPtrInSuccLists = hdev_allData;
	hdev_csrSuccLists = hdev_allData + num_nodi+1;
	hdev_csrPesiArchi = hdev_csrSuccLists + num_archi;
	hdev_ResNodeValues1 = hdev_csrPesiArchi + num_archi;
	if ((configuration.algoritmo == ALGOR_EG0)) {
		hdev_ResNodeValues2 = hdev_ResNodeValues1 + num_nodi; }
	else { hdev_ResNodeValues2 = hdev_ResNodeValues1; }
	hdev_flag = hdev_ResNodeValues2 + num_nodi;
	hdev_csrDataArchiAux = hdev_flag+1; 

	CUDASAFE( cudaMemcpyToSymbol(dev_allData, &hdev_allData, sizeof(int *), 0, cudaMemcpyHostToDevice) , "cudaMemcpyToSymbol dev_allData");
	CUDASAFE( cudaMemcpyToSymbol(dev_csrPtrInSuccLists, &hdev_csrPtrInSuccLists, sizeof(int *), 0, cudaMemcpyHostToDevice) , "cudaMemcpyToSymbol dev_csrPtrInSuccLists");
	CUDASAFE( cudaMemcpyToSymbol(dev_csrSuccLists, &hdev_csrSuccLists, sizeof(int *), 0, cudaMemcpyHostToDevice) , "cudaMemcpyToSymbol dev_csrSuccLists");
	CUDASAFE( cudaMemcpyToSymbol(dev_csrPesiArchi, &hdev_csrPesiArchi, sizeof(int *), 0, cudaMemcpyHostToDevice) , "cudaMemcpyToSymbol dev_csrPesiArchi");
	CUDASAFE( cudaMemcpyToSymbol(dev_ResNodeValues1, &hdev_ResNodeValues1, sizeof(int *), 0, cudaMemcpyHostToDevice) , "cudaMemcpyToSymbol dev_ResNodeValues1");
	CUDASAFE( cudaMemcpyToSymbol(dev_ResNodeValues2, &hdev_ResNodeValues2, sizeof(int *), 0, cudaMemcpyHostToDevice) , "cudaMemcpyToSymbol dev_ResNodeValues2");
	CUDASAFE( cudaMemcpyToSymbol(dev_flag, &hdev_flag, sizeof(int *), 0, cudaMemcpyHostToDevice) , "cudaMemcpyToSymbol dev_flag");
	CUDASAFE( cudaMemcpyToSymbol(dev_csrDataArchiAux, &hdev_csrDataArchiAux, sizeof(int *), 0, cudaMemcpyHostToDevice) , "cudaMemcpyToSymbol dev_csrDataArchiAux");

	CUDASAFE( cudaMemset(hdev_ResNodeValues1, 0, 2*num_nodi*sizeof(int)) , "cudaMemset: hdev_ResNodeValues1e2[]");

	if ((configuration.algoritmo == ALGOR_EG) || (configuration.algoritmo == ALGOR_EG0)) {
		int num_trans_mem = num_nodi+1+(2*num_archi);   
		num_trans_mem += 2*(num_nodi+1);  //2*(num_nodi+1) per nodeFlags1,2
		CUDASAFE( cudaMalloc((void **)&hdev_transData, num_trans_mem*sizeof(int)) , "cudaMalloc: &hdev_transData[]");
		hdev_cscPtrInPredLists = hdev_transData;
		hdev_cscPredLists = hdev_transData + num_nodi+1;
		hdev_cscPesiArchiPred = hdev_cscPredLists + num_archi;
		hdev_nodeFlags1 = hdev_cscPesiArchiPred + num_archi;
		hdev_nodeFlags2 = hdev_nodeFlags1 + num_nodi +1;

		CUDASAFE( cudaMemcpyToSymbol(dev_transData, &hdev_transData, sizeof(int *), 0, cudaMemcpyHostToDevice) , "cudaMemcpyToSymbol dev_transData");
		CUDASAFE( cudaMemcpyToSymbol(dev_cscPtrInPredLists, &hdev_cscPtrInPredLists, sizeof(int *), 0, cudaMemcpyHostToDevice) , "cudaMemcpyToSymbol dev_cscPtrInPredLists");
		CUDASAFE( cudaMemcpyToSymbol(dev_cscPredLists, &hdev_cscPredLists, sizeof(int *), 0, cudaMemcpyHostToDevice) , "cudaMemcpyToSymbol dev_cscPredLists");
		CUDASAFE( cudaMemcpyToSymbol(dev_cscPesiArchiPred, &hdev_cscPesiArchiPred, sizeof(int *), 0, cudaMemcpyHostToDevice) , "cudaMemcpyToSymbol dev_cscPesiArchiPred");
		CUDASAFE( cudaMemcpyToSymbol(dev_nodeFlags1, &hdev_nodeFlags1, sizeof(int *), 0, cudaMemcpyHostToDevice) , "cudaMemcpyToSymbol dev_nodeFlags1");
		CUDASAFE( cudaMemcpyToSymbol(dev_nodeFlags2, &hdev_nodeFlags2, sizeof(int *), 0, cudaMemcpyHostToDevice) , "cudaMemcpyToSymbol dev_nodeFlags2");

		CUDASAFE( cudaMemset(hdev_nodeFlags1, 0, 2*(1+num_nodi)*sizeof(int)) , "cudaMemset: hdev_ResNodeValues1e2[]");
	}

	return(0);
}

int dealloca_memoria_host() {
	int inn; 

	CUDASAFE( cudaFreeHost(host_allData) , "cudaFreeHost: host_allData[]");
	free(host_transData);
	free(csrPtrInSuccLists);
	free(nodePriority);
	free(nodeOwner);
	free(nodeFlags);
	free(csrPesiArchi);
	free(csrSuccLists);
	free(nomeExt_of_nomeInt);
	free(nomeInt_of_nomeExt);
	free(mapping);
	free(revmapping);
	for (inn=0; inn<num_nodi; inn++) { free(nodeName[inn]); }
	free(nodeName);

	return(0);
}

int dealloca_memoria_device() {
	cudaDeviceSynchronize();
        CUDASAFE( cudaFree(hdev_allData) , "cudaFree: &hdev_allData[]");
	return(0);
}



#define FLAGDEV_SOLVER_CU 1
#endif
