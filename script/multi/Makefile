
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
	rm -f *.o mkladder mkclique multi parser.h parser.y.c parser.lex parser.lex.c 


mkladder: mkladder.c
	gcc -O3 mkladder.c -o mkladder

mkclique: mkclique.c
	gcc -O3 mkclique.c -o mkclique


istanze: mkladder mkclique multi
	# clique-like games
	./mkclique 500 | gzip > clique_500.txt.gz
	./multi --copies 2 --input clique_500.txt.gz | gzip > clique_500_cp2.txt.gz
	./multi --copies 4 --input clique_500.txt.gz | gzip > clique_500_cp4.txt.gz
	./multi --copies 8 --input clique_500.txt.gz | gzip > clique_500_cp8.txt.gz
	./multi --copies 16 --input clique_500.txt.gz | gzip > clique_500_cp16.txt.gz
	./multi --copies 32 --input clique_500.txt.gz | gzip > clique_500_cp32.txt.gz
	./multi --copies 64 --input clique_500.txt.gz | gzip > clique_500_cp64.txt.gz
	./mkclique 1000 | gzip > clique_1000.txt.gz
	./multi --copies 2 --input clique_1000.txt.gz | gzip > clique_1000_cp2.txt.gz
	./multi --copies 4 --input clique_1000.txt.gz | gzip > clique_1000_cp4.txt.gz
	./multi --copies 8 --input clique_1000.txt.gz | gzip > clique_1000_cp8.txt.gz
	./multi --copies 16 --input clique_1000.txt.gz | gzip > clique_1000_cp16.txt.gz
	./multi --copies 32 --input clique_1000.txt.gz | gzip > clique_1000_cp32.txt.gz
	./multi --copies 64 --input clique_1000.txt.gz | gzip > clique_1000_cp64.txt.gz
	./mkclique 2000 | gzip > clique_2000.txt.gz
	./multi --copies 2 --input clique_2000.txt.gz | gzip > clique_2000_cp2.txt.gz
	./multi --copies 4 --input clique_2000.txt.gz | gzip > clique_2000_cp4.txt.gz
	./multi --copies 8 --input clique_2000.txt.gz | gzip > clique_2000_cp8.txt.gz
	./multi --copies 16 --input clique_2000.txt.gz | gzip > clique_2000_cp16.txt.gz
	./multi --copies 32 --input clique_2000.txt.gz | gzip > clique_2000_cp32.txt.gz
	./multi --copies 64 --input clique_2000.txt.gz | gzip > clique_2000_cp64.txt.gz
	# ladder-like games
	./mkladder 10000 | gzip > ladder_10000.txt.gz
	./multi --copies 2 --input ladder_10000.txt.gz | gzip > ladder_10000_cp2.txt.gz
	./multi --copies 4 --input ladder_10000.txt.gz | gzip > ladder_10000_cp4.txt.gz
	./multi --copies 8 --input ladder_10000.txt.gz | gzip > ladder_10000_cp8.txt.gz
	./multi --copies 16 --input ladder_10000.txt.gz | gzip > ladder_10000_cp16.txt.gz
	./multi --copies 32 --input ladder_10000.txt.gz | gzip > ladder_10000_cp32.txt.gz
	./mkladder 20000 | gzip > ladder_20000.txt.gz
	./multi --copies 2 --input ladder_20000.txt.gz | gzip > ladder_20000_cp2.txt.gz
	./multi --copies 4 --input ladder_20000.txt.gz | gzip > ladder_20000_cp4.txt.gz
	./multi --copies 8 --input ladder_20000.txt.gz | gzip > ladder_20000_cp8.txt.gz
	./multi --copies 16 --input ladder_20000.txt.gz | gzip > ladder_20000_cp16.txt.gz
	./multi --copies 32 --input ladder_20000.txt.gz | gzip > ladder_20000_cp32.txt.gz
	./mkladder 30000 | gzip > ladder_30000.txt.gz
	./multi --copies 2 --input ladder_30000.txt.gz | gzip > ladder_30000_cp2.txt.gz
	./multi --copies 4 --input ladder_30000.txt.gz | gzip > ladder_30000_cp4.txt.gz
	./multi --copies 8 --input ladder_30000.txt.gz | gzip > ladder_30000_cp8.txt.gz
	./multi --copies 16 --input ladder_30000.txt.gz | gzip > ladder_30000_cp16.txt.gz
	./multi --copies 32 --input ladder_30000.txt.gz | gzip > ladder_30000_cp32.txt.gz


TIMEOUT = 1800

run_esperimenti:
	# clique-like games
	./egsolver --timeout $(TIMEOUT) --cpu --input clique_500.txt.gz  > clique_500.out_cpu
	./egsolver --timeout $(TIMEOUT) --gpu --input clique_500.txt.gz  > clique_500.out_gpu
	./egsolver --timeout $(TIMEOUT) --cpu --input clique_500_cp2.txt.gz  > clique_500_cp2.out_cpu
	./egsolver --timeout $(TIMEOUT) --gpu --input clique_500_cp2.txt.gz  > clique_500_cp2.out_gpu
	./egsolver --timeout $(TIMEOUT) --cpu --input clique_500_cp4.txt.gz  > clique_500_cp4.out_cpu
	./egsolver --timeout $(TIMEOUT) --gpu --input clique_500_cp4.txt.gz  > clique_500_cp4.out_gpu
	./egsolver --timeout $(TIMEOUT) --cpu --input clique_500_cp8.txt.gz  > clique_500_cp8.out_cpu
	./egsolver --timeout $(TIMEOUT) --gpu --input clique_500_cp8.txt.gz  > clique_500_cp8.out_gpu
	./egsolver --timeout $(TIMEOUT) --cpu --input clique_500_cp16.txt.gz  > clique_500_cp16.out_cpu
	./egsolver --timeout $(TIMEOUT) --gpu --input clique_500_cp16.txt.gz  > clique_500_cp16.out_gpu
	./egsolver --timeout $(TIMEOUT) --cpu --input clique_1000.txt.gz  > clique_1000.out_cpu
	./egsolver --timeout $(TIMEOUT) --gpu --input clique_1000.txt.gz  > clique_1000.out_gpu
	./egsolver --timeout $(TIMEOUT) --cpu --input clique_1000_cp2.txt.gz  > clique_1000_cp2.out_cpu
	./egsolver --timeout $(TIMEOUT) --gpu --input clique_1000_cp2.txt.gz  > clique_1000_cp2.out_gpu
	./egsolver --timeout $(TIMEOUT) --cpu --input clique_1000_cp4.txt.gz  > clique_1000_cp4.out_cpu
	./egsolver --timeout $(TIMEOUT) --gpu --input clique_1000_cp4.txt.gz  > clique_1000_cp4.out_gpu
	./egsolver --timeout $(TIMEOUT) --cpu --input clique_1000_cp8.txt.gz  > clique_1000_cp8.out_cpu
	./egsolver --timeout $(TIMEOUT) --gpu --input clique_1000_cp8.txt.gz  > clique_1000_cp8.out_gpu
	./egsolver --timeout $(TIMEOUT) --cpu --input clique_1000_cp16.txt.gz  > clique_1000_cp16.out_cpu
	./egsolver --timeout $(TIMEOUT) --gpu --input clique_1000_cp16.txt.gz  > clique_1000_cp16.out_gpu
	./egsolver --timeout $(TIMEOUT) --cpu --input clique_2000.txt.gz  > clique_2000.out_cpu
	./egsolver --timeout $(TIMEOUT) --gpu --input clique_2000.txt.gz  > clique_2000.out_gpu
	./egsolver --timeout $(TIMEOUT) --cpu --input clique_2000_cp2.txt.gz  > clique_2000_cp2.out_cpu
	./egsolver --timeout $(TIMEOUT) --gpu --input clique_2000_cp2.txt.gz  > clique_2000_cp2.out_gpu
	./egsolver --timeout $(TIMEOUT) --cpu --input clique_2000_cp4.txt.gz  > clique_2000_cp4.out_cpu
	./egsolver --timeout $(TIMEOUT) --gpu --input clique_2000_cp4.txt.gz  > clique_2000_cp4.out_gpu
	./egsolver --timeout $(TIMEOUT) --cpu --input clique_2000_cp8.txt.gz  > clique_2000_cp8.out_cpu
	./egsolver --timeout $(TIMEOUT) --gpu --input clique_2000_cp8.txt.gz  > clique_2000_cp8.out_gpu
	./egsolver --timeout $(TIMEOUT) --cpu --input clique_2000_cp16.txt.gz  > clique_2000_cp16.out_cpu
	./egsolver --timeout $(TIMEOUT) --gpu --input clique_2000_cp16.txt.gz  > clique_2000_cp16.out_gpu
	#./egsolver --timeout $(TIMEOUT) --cpu --input clique_500_cp32.txt.gz  > clique_500_cp32.out_cpu
	#./egsolver --timeout $(TIMEOUT) --gpu --input clique_500_cp32.txt.gz  > clique_500_cp32.out_gpu
	#./egsolver --timeout $(TIMEOUT) --cpu --input clique_500_cp64.txt.gz  > clique_500_cp64.out_cpu
	#./egsolver --timeout $(TIMEOUT) --gpu --input clique_500_cp64.txt.gz  > clique_500_cp64.out_gpu
	#./egsolver --timeout $(TIMEOUT) --cpu --input clique_1000_cp32.txt.gz  > clique_1000_cp32.out_cpu
	#./egsolver --timeout $(TIMEOUT) --gpu --input clique_1000_cp32.txt.gz  > clique_1000_cp32.out_gpu
	#./egsolver --timeout $(TIMEOUT) --cpu --input clique_1000_cp64.txt.gz  > clique_1000_cp64.out_cpu
	#./egsolver --timeout $(TIMEOUT) --gpu --input clique_1000_cp64.txt.gz  > clique_1000_cp64.out_gpu
	#./egsolver --timeout $(TIMEOUT) --cpu --input clique_2000_cp32.txt.gz  > clique_2000_cp32.out_cpu
	#./egsolver --timeout $(TIMEOUT) --gpu --input clique_2000_cp32.txt.gz  > clique_2000_cp32.out_gpu
	#./egsolver --timeout $(TIMEOUT) --cpu --input clique_2000_cp64.txt.gz  > clique_2000_cp64.out_cpu
	#./egsolver --timeout $(TIMEOUT) --gpu --input clique_2000_cp64.txt.gz  > clique_2000_cp64.out_gpu
	# ladder-like games
	./egsolver --timeout $(TIMEOUT) --cpu --input ladder_10000.txt.gz  > ladder_10000.out_cpu
	./egsolver --timeout $(TIMEOUT) --gpu --input ladder_10000.txt.gz  > ladder_10000.out_gpu
	./egsolver --timeout $(TIMEOUT) --cpu --input ladder_10000_cp2.txt.gz  > ladder_10000_cp2.out_cpu
	./egsolver --timeout $(TIMEOUT) --gpu --input ladder_10000_cp2.txt.gz  > ladder_10000_cp2.out_gpu
	./egsolver --timeout $(TIMEOUT) --cpu --input ladder_10000_cp4.txt.gz  > ladder_10000_cp4.out_cpu
	./egsolver --timeout $(TIMEOUT) --gpu --input ladder_10000_cp4.txt.gz  > ladder_10000_cp4.out_gpu
	./egsolver --timeout $(TIMEOUT) --cpu --input ladder_10000_cp8.txt.gz  > ladder_10000_cp8.out_cpu
	./egsolver --timeout $(TIMEOUT) --gpu --input ladder_10000_cp8.txt.gz  > ladder_10000_cp8.out_gpu
	./egsolver --timeout $(TIMEOUT) --cpu --input ladder_20000.txt.gz  > ladder_20000.out_cpu
	./egsolver --timeout $(TIMEOUT) --gpu --input ladder_20000.txt.gz  > ladder_20000.out_gpu
	./egsolver --timeout $(TIMEOUT) --cpu --input ladder_20000_cp2.txt.gz  > ladder_20000_cp2.out_cpu
	./egsolver --timeout $(TIMEOUT) --gpu --input ladder_20000_cp2.txt.gz  > ladder_20000_cp2.out_gpu
	./egsolver --timeout $(TIMEOUT) --cpu --input ladder_20000_cp4.txt.gz  > ladder_20000_cp4.out_cpu
	./egsolver --timeout $(TIMEOUT) --gpu --input ladder_20000_cp4.txt.gz  > ladder_20000_cp4.out_gpu
	./egsolver --timeout $(TIMEOUT) --cpu --input ladder_20000_cp8.txt.gz  > ladder_20000_cp8.out_cpu
	./egsolver --timeout $(TIMEOUT) --gpu --input ladder_20000_cp8.txt.gz  > ladder_20000_cp8.out_gpu
	./egsolver --timeout $(TIMEOUT) --cpu --input ladder_30000.txt.gz  > ladder_30000.out_cpu
	./egsolver --timeout $(TIMEOUT) --gpu --input ladder_30000.txt.gz  > ladder_30000.out_gpu
	./egsolver --timeout $(TIMEOUT) --cpu --input ladder_30000_cp2.txt.gz  > ladder_30000_cp2.out_cpu
	./egsolver --timeout $(TIMEOUT) --gpu --input ladder_30000_cp2.txt.gz  > ladder_30000_cp2.out_gpu
	./egsolver --timeout $(TIMEOUT) --cpu --input ladder_30000_cp4.txt.gz  > ladder_30000_cp4.out_cpu
	./egsolver --timeout $(TIMEOUT) --gpu --input ladder_30000_cp4.txt.gz  > ladder_30000_cp4.out_gpu
	./egsolver --timeout $(TIMEOUT) --cpu --input ladder_30000_cp8.txt.gz  > ladder_30000_cp8.out_cpu
	./egsolver --timeout $(TIMEOUT) --gpu --input ladder_30000_cp8.txt.gz  > ladder_30000_cp8.out_gpu
#	./egsolver --timeout $(TIMEOUT) --cpu --input ladder_10000_cp16.txt.gz  > ladder_10000_cp16.out_cpu
#	./egsolver --timeout $(TIMEOUT) --gpu --input ladder_10000_cp16.txt.gz  > ladder_10000_cp16.out_gpu
#	./egsolver --timeout $(TIMEOUT) --cpu --input ladder_10000_cp32.txt.gz  > ladder_10000_cp32.out_cpu
#	./egsolver --timeout $(TIMEOUT) --gpu --input ladder_10000_cp32.txt.gz  > ladder_10000_cp32.out_gpu
#	./egsolver --timeout $(TIMEOUT) --cpu --input ladder_20000_cp16.txt.gz  > ladder_20000_cp16.out_cpu
#	./egsolver --timeout $(TIMEOUT) --gpu --input ladder_20000_cp16.txt.gz  > ladder_20000_cp16.out_gpu
#	./egsolver --timeout $(TIMEOUT) --cpu --input ladder_20000_cp32.txt.gz  > ladder_20000_cp32.out_cpu
#	./egsolver --timeout $(TIMEOUT) --gpu --input ladder_20000_cp32.txt.gz  > ladder_20000_cp32.out_gpu
#	./egsolver --timeout $(TIMEOUT) --cpu --input ladder_30000_cp16.txt.gz  > ladder_30000_cp16.out_cpu
#	./egsolver --timeout $(TIMEOUT) --gpu --input ladder_30000_cp16.txt.gz  > ladder_30000_cp16.out_gpu
#	./egsolver --timeout $(TIMEOUT) --cpu --input ladder_30000_cp32.txt.gz  > ladder_30000_cp32.out_cpu
#	./egsolver --timeout $(TIMEOUT) --gpu --input ladder_30000_cp32.txt.gz  > ladder_30000_cp32.out_gpu
