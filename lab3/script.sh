#!/bin/bash

#SBATCH --job-name=test_cluster				# Arbitrary job name
#SBATCH --output=TF_CPU_testbench_fruit.txt		# Output file labelling node
#SBATCH --time=01:00:00					# Max executin time
#SBATCH --partition=nano				# Slurm partition



echo "Starting slrum job"
srun python3 ./main2.py
echo "Finished slrum job"


