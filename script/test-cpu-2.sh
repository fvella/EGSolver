export OMP_NUM_THREADS=$1
../bin/egsolver --input clique_500_cp4.txt.gz  --eg0 --cpu --timeout 30 
