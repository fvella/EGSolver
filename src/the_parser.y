%{
#include <stdio.h>
#include <stdlib.h>
#include <zlib.h>
#include "egsolver.h"
#include "cpu_solver.h"

// stuff from flex that bison needs to know about:
int yylex();
int yyparse();
FILE *yyin;
extern gzFile my_gzfile;
void yyerror(const char *s);
extern int linenum;

double pow(double x, double y);
float powf(float x, float y);

void body_line_proc1(int identifier, int priority, int owner);


void mpg_arena_header_proc(int max);
void mpg_arena_footer_proc();
void mpg_arena_body_line_proc1(int identifier, int owner);
void mpg_arena_body_line_proc2();
void mpg_arena_thename_proc(char *name);
void mpg_arena_edge_proc(int succ, int w);
void mpg_arena_edge_procP(int succ, int w);

int auxdegreecount;

%}

// Bison fundamentally works by asking flex to get the next token, which it
// returns as an object of type "yystype".  But tokens could be of any
// arbitrary data type!  So we deal with that in Bison by defining a C union
// holding each of the types of tokens that Flex could return, and have Bison
// use that union instead of "int" for the definition of "yystype":
%union {
	int ival;
	float fval;
	char *sval;
}

// define the constant-string tokens:
%token ARENA
%token SEMICOL
%token COL
%token COMMA
%token MINUS
%token OWNERMAX
%token OWNERMIN

// define the "terminal symbol" token types I'm going to use (in CAPS
// by convention), and associate each with a field of the union:
%token <ival> INT
%token <fval> FLOAT
%token <sval> STRING

%start game

%%
game:
         mpg_arena
         ;




mpg_arena:
	mpg_arena_header mpg_arena_body_section { mpg_arena_footer_proc(); }
	;
mpg_arena_header:
	INT INT SEMICOL { mpg_arena_header_proc($1); } 
	| INT SEMICOL { mpg_arena_header_proc($1); } 
	| ARENA INT INT SEMICOL { mpg_arena_header_proc($2); } 
	| ARENA INT SEMICOL { mpg_arena_header_proc($2); } 
	;
mpg_arena_body_section:
	mpg_arena_body_lines
	;
mpg_arena_body_lines:
	mpg_arena_body_lines mpg_arena_body_line
	| mpg_arena_body_line
	;
mpg_arena_body_line:
	INT OWNERMAX { mpg_arena_body_line_proc1($1,0); } mpg_arena_edges mpg_arena_thename { mpg_arena_body_line_proc2(); }
	| INT OWNERMIN { mpg_arena_body_line_proc1($1,1); } mpg_arena_edges mpg_arena_thename { mpg_arena_body_line_proc2(); }
	;
mpg_arena_thename:
	STRING SEMICOL { mpg_arena_thename_proc($1); }
	| SEMICOL { mpg_arena_thename_proc(NULL); } 
	;
mpg_arena_edges:
	mpg_arena_edges COMMA mpg_arena_edge
	| mpg_arena_edge
	;

mpg_arena_edge:
	INT COL '-'INT  { mpg_arena_edge_proc($1,-($4)); }
	| INT COL INT  { mpg_arena_edge_proc($1,$3); }
	;

%%


void yyerror(const char *s) {
	fprintf(stderr,"Error reading input:  %s line %d\n", s, linenum);
	// might as well halt now:
	exit(-1);
}


/* funzioni per input di mgp nella forma
33;
0 MIN 1:1;
1 MIN 2:1,3:1,4:1;
2 MIN 5:1,6:1,7:1,8:1;
3 MAX 9:1,10:1,11:1,12:1;
4 MAX 1:1,7:1,11:1,13:1;
5 MIN 14:1,15:1,16:1,17:1,18:1;
...
dove nella prima linea c'e' il MASSIMO nodo. Ogni riga successiva
e' della forma
   node  owner  node:weight,...,node:weight  "string" ;
*/

void mpg_arena_header_proc(int max) {
	int idx;
//	printf("reading a game max id found: %d (at line %d)\n", max, linenum);
	input_gametype = GAMETYPE_MPG_ARENA; 
	counter_nodi = 0;
	counter_nodi0 = 0;
	counter_nodi0_1 = 0;
	counter_nodi0_2 = 0;
	max_nodo = max;
	num_nodi = max+1;
	num_archi = 0;
	num_pesi = 0;
	max_pesi = 0;
	min_outdegree = INT_MAX;
	max_outdegree = 0;
	numLowOutDegree[0] = numLowOutDegree[1] = numLowOutDegree[2] = numLowOutDegree[3] = 0;
	numLowInDegree[0] = numLowInDegree[1] = numLowInDegree[2] = numLowInDegree[3] = 0;
	memset(numAllInDegree, 0, LEN_DEGREEHIST*sizeof(int));
	memset(numAllOutDegree, 0, LEN_DEGREEHIST*sizeof(int));

	max_priority = 0;
	csrPtrInSuccLists = (int *)malloc((1+num_nodi)*sizeof(int)); checkNullAllocation(csrPtrInSuccLists,"allocazione csrPtrInSuccLists");
	nodePriority = (int *)malloc((1+num_nodi)*sizeof(int)); checkNullAllocation(nodePriority,"allocazione nodePriority");
	nodeOwner = (char *)malloc((1+num_nodi)*sizeof(char)); checkNullAllocation(nodeOwner,"allocazione nodeOwner");
	nodeName = (char **)malloc((1+num_nodi)*sizeof(char*)); checkNullAllocation(nodeName,"allocazione nodeName");

// TODO: temporaneamente allocato. Forse non serve....
	outDegrees_of_csr = (int *)malloc((1+num_nodi)*sizeof(int)); checkNullAllocation(outDegrees_of_csr,"allocazione outDegrees_of_csr");
	memset(outDegrees_of_csr, 0, (1+num_nodi)*sizeof(int));
	inDegrees_of_csr = (int *)malloc((1+num_nodi)*sizeof(int)); checkNullAllocation(inDegrees_of_csr,"allocazione inDegrees_of_csr");
	memset(inDegrees_of_csr, 0, (1+num_nodi)*sizeof(int));

	if ((configuration.algoritmo == ALGOR_EG) || (configuration.algoritmo == ALGOR_EG0)) {
		nodeFlags = (char *)malloc((1+num_nodi)*sizeof(char)); checkNullAllocation(nodeName,"allocazione nodeFlags");
		memset(nodeFlags, 0, (1+num_nodi)*sizeof(char));
	}

	size_csrSuccLists = 2*(1+num_nodi);  // almeno 2 archi uscenti per ogni nodo (poi si realloc, se serve)
	csrPesiArchi = (int *)malloc(size_csrSuccLists*sizeof(int)); checkNullAllocation(csrPesiArchi,"allocazione csrPesiArchi");
	csrSuccLists = (int *)malloc(size_csrSuccLists*sizeof(int)); checkNullAllocation(csrSuccLists,"allocazione csrSuccLists");

	nomeExt_of_nomeInt = (int *)malloc((1+num_nodi)*sizeof(int)); checkNullAllocation(nomeExt_of_nomeInt,"allocazione nomeExt_of_nomeInt");
	nomeInt_of_nomeExt = (int *)malloc((1+num_nodi)*sizeof(int)); checkNullAllocation(nomeInt_of_nomeExt,"allocazione nomeInt_of_nomeExt");
	for (idx=0; idx<(1+num_nodi); idx++) {
		nomeInt_of_nomeExt[idx] = -1;
	}
}

void body_line_proc1(int nomeEsterno, int priority, int owner) {
	auxdegreecount=num_archi;
	nomeExt_of_nomeInt[counter_nodi] = nomeEsterno;
	if (nomeEsterno <= max_nodo) {
		if (nomeInt_of_nomeExt[nomeEsterno] == -1) {
			nomeInt_of_nomeExt[nomeEsterno] = counter_nodi;
		} else {
			sprintf(str," %d (input line %d)", nomeEsterno, linenum);
			exitWithError("Error in input. Repeated node identifier: %s\n", str);
		}
	} else {
		sprintf(str,"%d greater that %d (input line %d)", nomeEsterno,max_nodo,linenum);
		exitWithError("Error in input. Unexpected node identifier: %s\n", str);
	}
	csrPtrInSuccLists[counter_nodi] = num_archi;
	nodePriority[counter_nodi] = priority;
	max_priority = MAX(max_priority, priority);
	nodeOwner[counter_nodi] = (char)owner;
	if ( (owner != 0) && (owner != 1) ) {
		sprintf(str," found %d changed to 1 (input line %d)", owner, linenum);
		printWarning("Warning. Node owner not in {0,1}: %s\n", str);
	}
}

void mpg_arena_body_line_proc2() {
	max_outdegree = MAX(max_outdegree, (num_archi-auxdegreecount));
	min_outdegree = MIN(min_outdegree, (num_archi-auxdegreecount));
	if ((num_archi-auxdegreecount)<4) {
		(numLowOutDegree[(num_archi-auxdegreecount)])++;
	}
	if ((num_archi-auxdegreecount)<(LEN_DEGREEHIST-1)) { (numAllOutDegree[(num_archi-auxdegreecount)])++;
	} else { (numAllOutDegree[LEN_DEGREEHIST-1])++; }
	
	if (nodeOwner[counter_nodi] == 0) {counter_nodi0++;};
	counter_nodi++;
}

void mpg_arena_thename_proc(char *name) {
	if (name == NULL) {
		nodeName[counter_nodi] = NULL;
	} else {
		nodeName[counter_nodi] = (char*)malloc((1+strlen(name))*sizeof(char)); checkNullAllocation(nodeName[counter_nodi],"allocazione nodeName[counter_nodi]");
		strcpy(nodeName[counter_nodi], name+1); //elimina quoting incluso dal parser
		nodeName[counter_nodi][strlen(name)-2] ='\0'; //elimina quoting incluso dal parser
	}
}



void mpg_arena_footer_proc() { 
	if (counter_nodi != num_nodi) {
		int i,j;
		fprintf(stderr, "counter_nodi=%d  num_nodi=%d\n", counter_nodi, num_nodi);
		for (i=0; i<num_nodi; i++){
			//fprintf(stderr, "i=%d\tnome=>%s<\n\t", i, nodeName[i]);
			fprintf(stderr, "i=%d\t\t", i);
			for (j=csrPtrInSuccLists[i]; j < csrPtrInSuccLists[i+1]; j++) { 
                                fprintf(stderr, " %d\t", csrSuccLists[j]);
                        }
                        fprintf(stderr, "\nindice            :\n"); for (i=0; i<(1+num_nodi); i++) { fprintf(stderr, "\t%d",i);} fprintf(stderr, "\n");
                        fprintf(stderr, "\nnomeInt_of_nomeExt:\n"); for (i=0; i<(1+num_nodi); i++) { fprintf(stderr, "\t%d",nomeInt_of_nomeExt[i]); } fprintf(stderr, "\n"); 
                        fprintf(stderr, "\nnomeExt_of_nomeInt:\n"); for (i=0; i<(1+num_nodi); i++) { fprintf(stderr, "\t%d",nomeExt_of_nomeInt[i]); } fprintf(stderr, "\n"); 
		}
		fflush(stderr);
		printf(str,"Expecting %d nodes, read %d nodes (input line %d)", num_nodi, counter_nodi,linenum);
		exitWithError("Error in input. %s\n", str);
	}
}






void mpg_arena_body_line_proc1(int nomeEsterno, int owner) {
// wrapping su body_line_proc1() fissando tutte le priority=0 (ora non sono usate, ma sprecano memoria)
	body_line_proc1(nomeEsterno, 0, owner);
}



void mpg_arena_edge_proc(int succ, int w) {
//TRACE fprintf(stdout,"%d:%d  ", succ, w);fflush(stdout);
	if (succ <= max_nodo) {
		if (num_archi >= (size_csrSuccLists-1)) {
			size_csrSuccLists*=2;
			csrPesiArchi = reallocazioneInt((ulong)size_csrSuccLists, csrPesiArchi);
			csrSuccLists = reallocazioneInt((ulong)size_csrSuccLists, csrSuccLists);
		}
		csrSuccLists[num_archi] = succ;
		csrPesiArchi[num_archi] = ((configuration.add_weights_mode == ADD_UNIT_WEIGHTS) ? 1
						: (((configuration.add_weights_mode == NOT_ADD_WEIGHTS) ? w
						       	: ( (configuration.add_weights_mode == ADD_RND_WEIGHTS) ? (configuration.rndWeightLow+(rand()%(1+configuration.rndWeightHigh-configuration.rndWeightLow)))
								 : (0) )))) ; 
		//csrPesiArchi[num_pesi] = w;
		//max_pesi = MAX(max_pesi,abs(w));
		max_pesi = MAX(max_pesi,abs(csrPesiArchi[num_archi]));
		num_archi++;
		//num_pesi++;
	} else {
		sprintf(str,"%d greater that %d (input line %d)", succ,max_nodo,linenum);
		exitWithError("Error in input. Unexpected successor identifier: %s\n", str);
	}
}

void mpg_arena_edge_procP(int succ, int w) {
	mpg_arena_edge_proc(succ, w);
//TRACE fprintf(stdout,"\n");fflush(stdout);
}






void parse_input() {
	do {    
		yyparse();
	} while (!gzeof(my_gzfile));
}

