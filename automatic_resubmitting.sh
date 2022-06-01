#!/bin/bash
# Submit a chain of batch jobs with dependencies
#
# Number of jobs to submit:
NR_OF_JOBS=20

# Batch job script:
JOB_SCRIPT=./bash_scripts/vspose_behavioral_best_comb_cpu_cobra.sh
echo "Submitting job chain of ${NR_OF_JOBS} jobs for batch script ${JOB_SCRIPT}:" >> job_chain.out
JOBID=$(sbatch ${JOB_SCRIPT} 2>&1 | awk '{print $(NF)}')
echo "  " ${JOBID} >> job_chain.out
I=1
while [[ ${I} -lt ${NR_OF_JOBS} ]]; do
JOBID=$(sbatch --dependency=afterany:${JOBID} ${JOB_SCRIPT} 2>&1 | awk '{print $(NF)}')
echo "  " ${JOBID} >> job_chain.out
let I=${I}+1
done
