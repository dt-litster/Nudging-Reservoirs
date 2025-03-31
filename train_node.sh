#!/bin/bash

declare -l SAVE_EXT="node_results/"

# declare -a WIDTHS=(10 50 100 5000)
# declare -a WINDOW_FNS=("5" "10" "20" "40"
#                        "min(10, t/500)" "min(20, t/500)" "min(40, t/500)"
#                        "max(5,5-5*np.cos(t*np.pi/2500))" "max(5,10-10*np.cos(t*np.pi/2500))" "max(5,20-20*np.cos(t*np.pi/2500))")

# declare -a LRS=("1e-3" "1e-4" "1e-5")
# declare -a WEIGHT_DECAYS=("1e-1" "1e-3" "1e-5")

# declare -i NUM_EPOCHS=30000
# declare -i SAVE_EVERY=6000

declare -a WIDTHS=(5000)
declare -a WINDOW_FNS=("40")

declare -a LRS=("1e-5")
declare -a WEIGHT_DECAYS=("1e-5")

declare -i NUM_EPOCHS=30000
declare -i SAVE_EVERY=6000

for WIDTH in "${WIDTHS[@]}"
do
  for WFN in "${WINDOW_FNS[@]}"
  do
    for LR in "${LRS[@]}"
    do
      for WD in "${WEIGHT_DECAYS[@]}"
      do
          # should probably send a command to slurm for each width or something.
          sbatch --export=ALL,SAVE_EXT="$SAVE_EXT",WIDTH="$WIDTH",NUM_EPOCHS="$NUM_EPOCHS",SAVE_EVERY="$SAVE_EVERY",WFN="$WFN",LR="$LR",WD="$WD" train_node_job_script.sh
          # I haven't coded anything to time things. Add --cuda or -c flag to run on cuda; otherwise, runs on CPU (should be fast enough on CPU anyway)
          # If you want to time things, should not use cuda to make comparable, but you should prbly be able just to take the time from SLURM receipt or something
      done
    done
  done
done
