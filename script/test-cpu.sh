export OMP_NUM_THREADS=$1
../bin/egsolver --input prova.mpg.gz --eg0 --cpu --solve --timeout 300 $EXTRA
