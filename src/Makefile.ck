
#scegliere la compute capability del target device:
#CAPABILITY = -arch=sm_20
#CAPABILITY = -arch=sm_21
#CAPABILITY = -arch=sm_30
#CAPABILITY = -arch=sm_35
CAPABILITY = ${CUDA_ARC}


CC = gcc
NVCC = nvcc
FLEX = flex
BISON = bison

#potrebbero servire anche   -L/usr/lib64   e    -lstdc++   o simili a seconda della distribuzione
LIBS = -L ${CUDA_HOME}/lib64 -lcuda -lcudart -lm  -lstdc++ -lcusparse -lz
INCLUDES= -I ${CUDA_HOME}/include

OBJECTS = errori.o  parser.lex.o  parser.y.o  sig_handling.o  dev_solver.o  cpu_solver.o host_csr2csc.o
CHEADERS = common.h  errori.h  egsolver.h  sig_handling.h cpu_solver.h host_csr2csc.h sorting_criteria.h
CUHEADERS = utils.h  dev_common.h thrust_wrapper.h 


BIN_DIR = ${DEST_DIR}

.PHONY: clean all help versioniShuffling

#Opzioni di compilazione per il gcc
CCFLAG = -Wall -Wextra -Wno-sign-compare -Wformat=2  -O3 -fopenmp
#CCFLAG = -Wall -Wextra -Wformat=2  -O3  
#CCFLAG = -Wall -Wformat=2  -O3  

#Opzioni di compilazione di gcc per il DEBUG 
#CCFLAG = -Wall -Wextra -Wformat=2 -g3 -pg -fbounds-check 


#Opzioni di compilazione per NVCC (GPU)
# Varie opzioni di compilazione per nvcc:
GFLAG = 
#GFLAG = -lineinfo
#GFLAG = --verbose
#GFLAG = -use_fast_math 
#GFLAG = -use_fast_math --ptxas-options="-v" -Xptxas -dlcm=ca
#GFLAG = -use_fast_math --maxrregcount 32 --ptxas-options="-v" -Xptxas -dlcm=ca
#GFLAG = -use_fast_math --maxrregcount 24 --ptxas-options="-v" -Xptxas -dlcm=ca
#GFLAG = -use_fast_math --maxrregcount 16 --ptxas-options="-v" -Xptxas -dlcm=ca
#GFLAG = -use_fast_math --maxrregcount 16 --ptxas-options="-v" 
#GFLAG = -use_fast_math --ptxas-options="-v" 

# Varie opzioni per il DEBUG con nvcc:
#GFLAG = -pg -g -G -lineinfo --compiler-options -Wall
#GFLAG = -pg -g -G -lineinfo --ptxas-options="-v" -Xptxas -dlcm=ca --compiler-options -Wall
#GFLAG = -pg -g -G --maxrregcount 16 --ptxas-options="-v" -Xptxas -dlcm=ca
#GFLAG = -pg -g -G --ptxas-options="-v"
#GFLAG = -pg -g -G --maxrregcount 32 --ptxas-options="-v" 
#GFLAG = -pg -g -G --maxrregcount 16 --ptxas-options="-v" 
#GFLAG = -pg -g -G 


help:
	@echo "use 'make egsolver' to generate $(BIN_DIR)/egsolver"
	@echo "use 'make all' to generate all"
	@echo "use 'make clean' to remove all temporary files and executables"

all: egsolver

egsolver: $(OBJECTS) egsolver.o
	$(CC) -o $(BIN_DIR)/egsolver egsolver.o $(OBJECTS) -lfl $(CCFLAG) $(LIBS)  $(INCLUDES)



dev_solver.o: dev_solver.cu dev_EG_alg.cu csr2csc.cu thrust_wrapper.cu utils.cu $(CUHEADERS) $(CHEADERS)
	$(NVCC)  -c dev_solver.cu utils.cu $(CAPABILITY) $(GFLAG)

parser.y.o: parser.y.c
	$(CC)  -c parser.y.c -o parser.y.o $(CCFLAG) $(INCLUDES)

parser.lex.o: parser.lex.c
	$(CC)  -c parser.lex.c -o parser.lex.o -O3 $(INCLUDES)

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


egsolver.o: egsolver.c $(CHEADERS) parser.y.c
	$(CC)  -c egsolver.c -o egsolver.o $(CCFLAG) $(INCLUDES)

errori.o: errori.c $(CHEADERS)
	$(CC)  -c errori.c -o errori.o $(CCFLAG) $(INCLUDES)

sig_handling.o: sig_handling.c $(CHEADERS)
	$(CC)  -c sig_handling.c -o sig_handling.o $(CCFLAG) $(INCLUDES)

host_csr2csc.o: host_csr2csc.c $(CHEADERS)
	$(CC)  -c host_csr2csc.c -o host_csr2csc.o $(CCFLAG) $(INCLUDES)

cpu_solver.o: cpu_solver.c $(CHEADERS)
	$(CC)  -c cpu_solver.c -o cpu_solver.o $(CCFLAG) $(INCLUDES)




clean:
	rm -f *.o parser.y parser.h parser.y.c parser.lex parser.lex.c  $(BIN_DIR)/egsolver


versionsShuffling:
	# making with shuffle 2:
	cp dev_EG_alg_shfl_full_2tpv.cu dev_EG_alg.cu
	touch dev_EG_alg.cu
	$(MAKE) egsolver 
	mv $(BIN_DIR)/egsolver $(BIN_DIR)/egsolver_shfl_full_2tpv
	# making with shuffle 4:
	cp dev_EG_alg_shfl_full_4tpv.cu dev_EG_alg.cu
	touch dev_EG_alg.cu
	$(MAKE) egsolver 
	mv $(BIN_DIR)/egsolver $(BIN_DIR)/egsolver_shfl_full_4tpv
	# making with shuffle 8:
	cp dev_EG_alg_shfl_full_8tpv.cu dev_EG_alg.cu
	touch dev_EG_alg.cu
	$(MAKE) egsolver 
	mv $(BIN_DIR)/egsolver $(BIN_DIR)/egsolver_shfl_full_8tpv
	# making with shuffle 16:
	cp dev_EG_alg_shfl_full_16tpv.cu dev_EG_alg.cu
	touch dev_EG_alg.cu
	$(MAKE) egsolver 
	mv $(BIN_DIR)/egsolver $(BIN_DIR)/egsolver_shfl_full_16tpv
	# making with shuffle 32:
	cp dev_EG_alg_shfl_full_32tpv.cu dev_EG_alg.cu
	touch dev_EG_alg.cu
	$(MAKE) egsolver 
	mv $(BIN_DIR)/egsolver $(BIN_DIR)/egsolver_shfl_full_32tpv
	# making without shuffle:
	cp dev_EG_alg_shfl_none.cu dev_EG_alg.cu
	touch dev_EG_alg.cu
	$(MAKE) egsolver 
	mv $(BIN_DIR)/egsolver $(BIN_DIR)/egsolver_shfl_none


