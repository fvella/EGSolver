
#ifndef FLAGLOADSIGHANDLER_C
#include "sig_handling.h"

void install_handler() {

  static struct sigaction act;
  act.sa_handler = my_catchint; /* registrazione dell'handler */

  sigfillset(&(act.sa_mask)); /* tutti i segnali saranno ignorati
                                 DURANTE l'esecuzione dell'handler */

  /* imposto l'handler per il segnale SIGINT */
  sigaction(SIGINT, &act, NULL); 
  sigaction(SIGUSR1, &act, NULL); 

}

void install_alarmhandler() {

  static struct sigaction act;
  act.sa_handler = my_catchalarm; /* registrazione dell'handler */

  sigfillset(&(act.sa_mask)); /* tutti i segnali saranno ignorati
                                 DURANTE l'esecuzione dell'handler */

  sigaction(SIGALRM, &act, NULL); 

}

 /* Questo e' l'handler. Semplice: stampa il segnale. */
void my_catchint(int signo) {
	//fprintf(stderr,"\nsegnale %d\n",signo);fflush(stderr);
	if ((signo==SIGINT) || (signo==SIGUSR1)) {
	char dumpfilename[32];
	//FILE *dumpfile;
	//extern parse out;

	sprintf(dumpfilename, "dump-data-%d", getpid());
	//printf("\nERXXXXXXXXXXXX-%s-XXXXXXXXXXX\n",dumpfilename);fflush(stdout);
	//if ((dumpfile=fopen(dumpfilename,"w")) == NULL) {
	printf("\nCATCHING SIG_INT: signo=%d... dumping data into file %s\n", signo,dumpfilename);
	if ((freopen(dumpfilename,"w", stdout)) == NULL) {
		fprintf(stderr,"\nERROR in opening file %s :", dumpfilename); perror(""); fprintf(stderr,"\n");
		exit(1);
	}
//	cudaMemcpy(out.host_ngb, out.devh_ngb, (out.allocato_ngb)*sizeof(int), cudaMemcpyDeviceToHost);
//	cudaMemcpy(out.host_nga, out.devh_nga, (out.allocato_nga)*sizeof(int), cudaMemcpyDeviceToHost);
	sleep(10);
	//fprintf(dumpfile,"\nERXXXXXXXXXXXXXXXXXXXXXXX\n");
	//fclose(dumpfile);
	exit(2);
	}
}


void my_catchalarm(int signo) {

	if ((signo==SIGALRM)) {
		timeout_expired = 1;
		fprintf(stderr,"\nTIMEOUT EXPIRED: forced exit\n");fflush(stderr);
		//exit(2);
	}
}


#define FLAGLOADSIGHANDLER_C 1
#endif
