%{
#include <stdio.h>
#include <limits.h>
#include "parser.h"
extern int my_abstractread(char *buff, int buffsize);


int a2int(const char *ptr) {
        char *endptr;
        long val;

        errno = 0;
        val = strtol(ptr, NULL, 10);

        if ((errno == ERANGE && (val == LONG_MAX || val == LONG_MIN)) || (errno != 0 && val == 0)) {
                fprintf(stderr,"\nCONVERSION ERROR illegal or out-of-range value in input\n\n");
                exit(EXIT_FAILURE);
        }
        if ((val < INT_MIN) || (val > INT_MAX)) {
                fprintf(stderr,"\nCONVERSION ERROR out-of-range [%d,%d] value in input: %ld\n\n", INT_MIN, INT_MAX, val);
                exit(EXIT_FAILURE);
        }
        return((int)val);
}


#define YY_INPUT(buff, res, buffsize) (res = my_abstractread(buff, buffsize))
#define YY_DECL int yylex()

int linenum = 1;
%}
%%
[ \t] ;
\n    { ++linenum; }
arena           { return ARENA; }
MAX           { return OWNERMAX; }
MIN           { return OWNERMIN; }
[;]           { return SEMICOL; }
[:]           { return COL; }
[,]           { return COMMA; }
[-]           { return '-'; }
"#".*        ; // { printf("This is a comment\n"); }
"%".*        ; // { printf("This is a comment\n"); }
"//".*       ; //  { printf("This is a comment\n"); }
[0-9]+\.[0-9]+ { yylval.fval = atof(yytext); return FLOAT; }
[0-9]+         { yylval.ival = a2int(yytext); return INT; }
\"(\\.|[^"])*\"   {
	// we have to copy because we can't rely on yytext not changing underneath us:
	yylval.sval = strdup(yytext);
	return STRING;
}
.              ;
%%
  // int main(int argc, char** argv) {
  // 	// lex through the input:
  // 	yylex();
  // 	return(0);
  // }
