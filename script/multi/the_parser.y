%{
#include <stdio.h>
#include <stdlib.h>
#include <zlib.h>
#include "multi.h"

// stuff from flex that bison needs to know about:
int yylex();
int yyparse();
FILE *yyin;
extern gzFile my_gzfile;
void yyerror(const char *s);
extern int linenum;

double pow(double x, double y);
float powf(float x, float y);



void mpg_arena_header_proc(int max);
void mpg_arena_footer_proc();
void mpg_arena_body_line_proc1(int identifier, int owner);
void mpg_arena_body_line_proc2();
void mpg_arena_thename_proc(char *name);
void mpg_arena_edge_proc(int succ, int w);


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
	mpg_arena_edges COMMA {fprintf(stdout,","); } mpg_arena_edge
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
	numSucc=0;
	lastNode = -1;
	lastOwner = -1;

	max_nodo = max;
	//fprintf(stdout,"[numCopie:%d]", numCopie);
	multi_max_nodo = max+((numCopie-1)*(max+1));
	fprintf(stdout,"%d;\n", multi_max_nodo);
}

void mpg_arena_body_line_proc2() {
	;
}

void mpg_arena_thename_proc(char *name) {
	int i,j;
	//fprintf(stdout,"[numSucc:%d]", numSucc);
for (j=1; j<numCopie; j++) {
	if (numSucc>0) {
		fprintf(stdout,",%d:%d", successori[0]+(j*(1+max_nodo)), pesi[0]);
		for (i=1; i<numSucc; i++) {
			fprintf(stdout,",%d:%d", successori[i]+(j*(1+max_nodo)), pesi[i]);
		}
	}
}
	if (name != NULL) { fprintf(stdout,"; %s\n", name);
	} else { fprintf(stdout,";\n"); }

for (j=1; j<numCopie; j++) {
	fprintf(stdout,"%d %s ", lastNode+(j*(1+max_nodo)), (lastOwner==0?"MAX":"MIN"));
	if (numSucc>0) {
		fprintf(stdout,"%d:%d", successori[0]+(j*(1+max_nodo)), pesi[0]);
		for (i=1; i<numSucc; i++) {
			fprintf(stdout,",%d:%d", successori[i]+(j*(1+max_nodo)), pesi[i]);
		}
	}
	if (name != NULL) { fprintf(stdout,"; %sX\n", name);
	} else { fprintf(stdout,";\n"); }
}
	
	numSucc=0;
	lastNode = -1;
	lastOwner = -1;
}

void mpg_arena_footer_proc() { 
	;
}

void mpg_arena_body_line_proc1(int nomeEsterno, int owner) {
	fprintf(stdout,"%d %s ", nomeEsterno, (owner==0?"MAX":"MIN"));
	lastNode = nomeEsterno;
	lastOwner = owner;
}

void mpg_arena_edge_proc(int succ, int w) {
	successori[numSucc] = succ;
	pesi[numSucc++] = w;
	fprintf(stdout,"%d:%d", succ, w);
	//fprintf(stdout,"%d:%d", 1+max_nodo+succ, w);
}

void parse_input() {
	do {    
		yyparse();
	} while (!gzeof(my_gzfile));
}

