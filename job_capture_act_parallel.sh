#!/bin/bash
#SBATCH -n 1                          # 1 CPU core per job
#SBATCH --array=1-18%18               # Submit 30 jobs, with up to 10 running concurrently
#SBATCH -t 02:30:00                   # Time limit: 2 hour 30 minutes per job
#SBATCH --partition=use-everything    # Specify the partition to run on
#SBATCH --output=logs/job_avgpool_vanilla_re_%A_%a.out    # Output logs saved in logs/ as job_avgpool_vanilla_<JobID>_<TaskID>.out

echo "Starting job for model index $SLURM_ARRAY_TASK_ID on node $(hostname)"
source bin/activate
python capture_act_parallel.py $SLURM_ARRAY_TASK_ID
echo "Finished job for model index $SLURM_ARRAY_TASK_ID"