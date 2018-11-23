
CC = gcc
FLEX = flex
BISON = bison

LIBS = -lm  -lstdc++ -lz
INCLUDES=  

.PHONY: clean all

#Opzioni di compilazione per il gcc
CCFLAG = -Wall -Wextra -Wno-sign-compare -Wformat=2  -O3 -fopenmp
#CCFLAG = -Wall -Wextra -Wformat=2  -O3  
#CCFLAG = -Wall -Wformat=2  -O3  

#Opzioni di compilazione di gcc per il DEBUG 
#CCFLAG = -Wall -Wextra -Wformat=2 -g3 -pg -fbounds-check 


all: multi.c parser.y.c common.h  errori.c  errori.h  parser.lex.c
	$(CC) -o multi multi.c errori.c parser.y.c parser.lex.c -lfl $(CCFLAG) $(LIBS)  $(INCLUDES)


parser.y.c: parser.y parser.lex the_lex.l
	$(BISON) -d parser.y
	mv -f parser.tab.c parser.y.c
	mv -f parser.tab.h parser.h

parser.lex.c: parser.lex the_lex.l parser.y.c
	$(FLEX) parser.lex
	mv lex.yy.c -f parser.lex.c

parser.y: the_parser.y
	cp -f the_parser.y parser.y

parser.lex: the_lex.l
	cp -f the_lex.l parser.lex


clean:
	rm -f *.o multi parser.h parser.y.c parser.lex parser.lex.c 




