/** \file  egsolver.h
 * \brief Header-file per tutto il solver
 *
 */

#ifndef FLAGMPGSOLVER_H

#include "common.h"
#include "errori.h"
#include "parser.h"
#include "sig_handling.h"

#include <assert.h>

// da #include "dev_solver.h"
extern int alloca_memoria_host();
extern int alloca_memoria_device();
extern int dealloca_memoria_host();
extern int dealloca_memoria_device();
extern int copia_dati_su_device();
extern void gpu_solver();


char str[256];
int *nomeExt_of_nomeInt;
int *nomeInt_of_nomeExt;
int *mapping;
int *revmapping;

config configuration;
stat statistics;
// spostato in struct configuration: char filename[FILENAME_LEN];
//GZINPUT  FILE *inputsource;
uint timeout_expired;

// CSR zero-based (i nodi sono interi e 0 e' un possibile nodo)
int counter_nodi;         /**< numero di nodi letti */
int counter_nodi0;        /**< numero di nodi letti */
int max_nodo;             /**< massimo numero/nodo (i nodi sono gli interi 0..num_nodi) */
int min_outdegree;        /**< minimo numero di successori di un nodo */
int max_outdegree;        /**< massimo numero di successori di un nodo */
double avg_outdegree;     /**< numero medio di successori di un nodo */
double stddev_outdegree;  /**< std dev su num di successori di un nodo */
int numLowOutDegree[4];   /**< numero di nodi con outdegree pari a 0 1 2 3 (In realta' si suppone che outdegree sia sempre >0) */
int numLowInDegree[4];    /**< numero di nodi con  indegree pari a 0 1 2 3 (In realta' si suppone che indegree sia sempre >0) */
int numAllInDegree[LEN_DEGREEHIST];  /**< numero di nodi con  indegree pari a 0 .. 126, >126  */
int numAllOutDegree[LEN_DEGREEHIST];  /**< numero di nodi con outdegree pari a 0 .. 126, >126  */
int num_nodi;             /**< max_nodo+1  (i nodi sono gli interi 0..num_nodi) */
int num_archi;            /**< numero di archi (nnz: nonzero elements) */
int max_pesi;             /**< peso massimo (in valore abs) */
int MG_pesi;              /**< soglia MG dell'algoritmo EG */
int num_pesi;             /**< numero di pesi (alla fine di ogni riga letta coinciderca' con num_archi) */
int *csrPesiArchi;        /**< pesi degli archi per num_archi archi */
int size_csrSuccLists;    /**< spazio attualmente allocato per csrPesiArchi[] (e csrSuccLists[]) */
int *csrPtrInSuccLists;   /**< num_nodi+1 elementi che puntano in csrPesiArchi[] e csrSuccLists[] */
int *csrSuccLists;        /**< nodi successori negli num_archi archi */
int *revSuccLists;        /**< revSuccLists[i] e' il nodo origine dell'arco che giunge a csrSuccLists[i] (num_archi elementi) */

int *outDegrees_of_csr;
int *inDegrees_of_csr;

int *nomeExt_of_nomeInt;  /**< num_nodi+1 elementi che memorizzano gli id esterni dei nodi */
int *nomeInt_of_nomeExt;  /**< relazione inversa di nomeExt_of_nomeInt[] */

int max_priority;         /**< priorita massima */
int *nodePriority;        /**< num_nodi+1 elementi, nodePriority[i] = priorita' dell'i-esimo nodo letto (NON del nodo i) */
char *nodeOwner;          /**< num_nodi+1 elementi, nodeOwner[i] = owner dell'i-esimo nodo letto */
char *nodeFlags;          /**< num_nodi+1 flag boolean usato da alcuni algor per partizionare i nodi */

char **nodeName;          /**< num_nodi+1 elementi, nodeName[i][] = name dell'i-esimo nodo letto */

int input_gametype;
int output_gametype;


int *host_allData;            /**< blocco unico che contiene i tre successici array */
int *host_csrPtrInSuccLists;  /**< num_nodi+1 elementi che puntano in csrPesiArchi[] e csrSuccLists[] */
int *host_csrSuccLists;       /**< nodi successori negli num_archi archi (num_archi elementi) */
int *host_revSuccLists;       /**< revSuccLists[i] e' il nodo origine dell'arco che giunge a csrSuccLists[i] (num_archi elementi) */
int *host_csrPesiArchi;       /**< pesi degli archi per num_archi archi (num_archi elementi) */
int *host_ResNodeValues1;      /**< risultato, il v(i) per ogni nodo (num_nodi elementi) */
int *host_ResNodeValues2;      /**< risultato, il v(i) per ogni nodo (num_nodi elementi), all'ultimo o penultimo passo */
int *host_ResNodeValuesAux;    /**< ausilio per inizializzazione di host_ResNodeValues1[] in alcuni algoritmi (ZP,...) */
int *host_flag;                /**< flag di controllo per loop di kernel */

int *host_csrDataArchiAux;       /**< array auriliario di num_archi elementi (compara) */

// definito in cs_common:   int *dev_allData;             // corrispondente a host_allData su device
// definito in cs_common:   int *dev_csrPtrInSuccLists;   // corrispondente a host_csrPtrInSuccLists su device
// definito in cs_common:   int *dev_csrSuccLists;        // corrispondente a host_csrSuccLists su device
// definito in cs_common:   int *dev_csrPesiArchi;        // corrispondente a host_csrPesiArchi su device
// definito in cs_common:   int *dev_ResNodeValues1       // corrispondente a host_NodeValues su device
// definito in cs_common:   int *dev_ResNodeValues2       // corrispondente a host_NodeValues su device
// definito in cs_common:   int *dev_ResNodeValuesAux     // corrispondente a host_NodeValues su device

int *hdev_allData;             /**< var host che memorizza ptr-host di dev_allData su device */
int *hdev_csrPtrInSuccLists;   /**< idem per dev_csrPtrInSuccLists */
int *hdev_csrSuccLists;        /**< idem per dev_csrSuccLists */
int *hdev_revSuccLists;        /**< idem per dev_revSuccLists */
int *hdev_csrPesiArchi;        /**< idem per dev_csrPesiArchi */
int *hdev_ResNodeValues1;       /**< idem per dev_ResNodeValues1 */
int *hdev_ResNodeValues2;       /**< idem per dev_ResNodeValues2 */
int *hdev_ResNodeValuesAux;     /**< idem per dev_ResNodeValuesAux */
int *hdev_flag;                 /**< idem per dev_flag */
int *hdev_nodeFlags1;          /**< num_nodi+1 flag (int) */
int *hdev_nodeFlags2;          /**< num_nodi+1 flag (int) */

int *host_transData;            /**< blocco unico che contiene i tre successici array */
int *host_cscPtrInPredLists;   /**< copia host (se usata) di dev_cscPtrInPredLists */
int *host_cscPredLists;        /**< copia host (se usata) di dev_cscPredLists */
int *host_cscPesiArchiPred;    /**< copia host (se usata) di dev_cscPesiArchiPred */
int *hdev_transData;            /**< var host che memorizza ptr-host di dev_transData su device */
int *hdev_cscPtrInPredLists;   /**< corrispondenre CSC della CSR dev_csrPtrInSuccLists */
int *hdev_cscPredLists;        /**< corrispondenre CSC della CSR dev_csrSuccLists */
int *hdev_cscPesiArchiPred;    /**< corrispondenre CSC della CSR dev_csrPesiArchi */

int *hdev_csrDataArchiAux;       /**< array auriliario di num_archi elementi (compara) */

/** \brief Wrapper al codice per il parsing
 *
 **/
void parse_input();


/** \brief Riordina (eventualmente rinomina i nodi) usando interi tra 0 e num_nodi
 *
 * \todo Ridefinire le priority (se il caso) come interi tra 1 e max_priority
 **/
void remap_instance();


/** \brief Stampa l'istanza MPG trasformata come da opzioni (usando il mapping calcolato da remap_instance())
 *
 **/
void output_translation();


/** \brief Predispone i dati per la computazione della soluzione
 *
 * \details Alloca la memoria sia host che device (pinned)
 * \n copia i dati in memoria host usando il mapping calcolato da remap_instance()
 *
 **/
void postparsing();


/** \brief Processa la command-line e gestisce i parametri di default
 *
 **/
void setconfig(int argc, char *argv[]);


/** \brief Inizializza statistiche
 *
 **/
void setstat();


void checkExistsParameter (int i, int argc, char * msg, char **argv);

int checkExistsOptionalParameter (int i, int argc, char **argv);

int myatoi(char *str, char *msg, char **argv);



long aggiorna_max_loop(long Narchi, long Nnodi, long MGpesi, long maxloop);





#define FLAGMPGSOLVER_H 1
#endif
