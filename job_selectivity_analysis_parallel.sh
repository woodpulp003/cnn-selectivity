#!/bin/bash
#SBATCH -n 1
#SBATCH --array=1-48%48  # Submit 18 jobs (one per model), with at most 18 running concurrently
#SBATCH -t 03:00:00
#SBATCH --output=logs/selectivity_avgpool_%A_%a.out
#SBATCH --error=logs/selectivity_avgpool_%A_%a.err

source bin/activate

# Change to myenv directory
cd /om2/user/sahil003/myenv

# List of activation directories for each model.
# Each model's dump (i.e., all 6 layer files) is stored in one directory.
ACTIVATION_DIRS=(
    # -------------------------
    # Varying Face
    # -------------------------
    /mindhive/nklab3/users/sahil003/activation_dump_avgpool/varying_face_0%_v1
    /mindhive/nklab3/users/sahil003/activation_dump_avgpool/varying_face_0%_v2
    /mindhive/nklab3/users/sahil003/activation_dump_avgpool/varying_face_0%_v3

    /mindhive/nklab3/users/sahil003/activation_dump_avgpool/varying_face_33%_v1
    /mindhive/nklab3/users/sahil003/activation_dump_avgpool/varying_face_33%_v2
    /mindhive/nklab3/users/sahil003/activation_dump_avgpool/varying_face_33%_v3

    /mindhive/nklab3/users/sahil003/activation_dump_avgpool/varying_face_100%_v1
    /mindhive/nklab3/users/sahil003/activation_dump_avgpool/varying_face_100%_v2
    /mindhive/nklab3/users/sahil003/activation_dump_avgpool/varying_face_100%_v3

    # -------------------------
    # Varying Food
    # -------------------------
    /mindhive/nklab3/users/sahil003/activation_dump_avgpool/varying_food_0%_v1
    /mindhive/nklab3/users/sahil003/activation_dump_avgpool/varying_food_0%_v2
    /mindhive/nklab3/users/sahil003/activation_dump_avgpool/varying_food_0%_v3

    /mindhive/nklab3/users/sahil003/activation_dump_avgpool/varying_food_33%_v1
    /mindhive/nklab3/users/sahil003/activation_dump_avgpool/varying_food_33%_v2
    /mindhive/nklab3/users/sahil003/activation_dump_avgpool/varying_food_33%_v3

    /mindhive/nklab3/users/sahil003/activation_dump_avgpool/varying_food_100%_v1
    /mindhive/nklab3/users/sahil003/activation_dump_avgpool/varying_food_100%_v2
    /mindhive/nklab3/users/sahil003/activation_dump_avgpool/varying_food_100%_v3

    # -------------------------
    # Cutout Color Final (v1–v10)
    # -------------------------
    /mindhive/nklab3/users/sahil003/activation_dump_avgpool/cutout_color_final_v1
    /mindhive/nklab3/users/sahil003/activation_dump_avgpool/cutout_color_final_v2
    /mindhive/nklab3/users/sahil003/activation_dump_avgpool/cutout_color_final_v3
    /mindhive/nklab3/users/sahil003/activation_dump_avgpool/cutout_color_final_v4
    /mindhive/nklab3/users/sahil003/activation_dump_avgpool/cutout_color_final_v5
    /mindhive/nklab3/users/sahil003/activation_dump_avgpool/cutout_color_final_v6
    /mindhive/nklab3/users/sahil003/activation_dump_avgpool/cutout_color_final_v7
    /mindhive/nklab3/users/sahil003/activation_dump_avgpool/cutout_color_final_v8
    /mindhive/nklab3/users/sahil003/activation_dump_avgpool/cutout_color_final_v9
    /mindhive/nklab3/users/sahil003/activation_dump_avgpool/cutout_color_final_v10

    # -------------------------
    # Mixed Final (v1–v10)
    # -------------------------
    /mindhive/nklab3/users/sahil003/activation_dump_avgpool/mixed_final_v1
    /mindhive/nklab3/users/sahil003/activation_dump_avgpool/mixed_final_v2
    /mindhive/nklab3/users/sahil003/activation_dump_avgpool/mixed_final_v3
    /mindhive/nklab3/users/sahil003/activation_dump_avgpool/mixed_final_v4
    /mindhive/nklab3/users/sahil003/activation_dump_avgpool/mixed_final_v5
    /mindhive/nklab3/users/sahil003/activation_dump_avgpool/mixed_final_v6
    /mindhive/nklab3/users/sahil003/activation_dump_avgpool/mixed_final_v7
    /mindhive/nklab3/users/sahil003/activation_dump_avgpool/mixed_final_v8
    /mindhive/nklab3/users/sahil003/activation_dump_avgpool/mixed_final_v9
    /mindhive/nklab3/users/sahil003/activation_dump_avgpool/mixed_final_v10

    # -------------------------
    # Noncutout Greyscale (v1–v10)
    # -------------------------
    /mindhive/nklab3/users/sahil003/activation_dump_avgpool/noncutout_greyscale_v1
    /mindhive/nklab3/users/sahil003/activation_dump_avgpool/noncutout_greyscale_v2
    /mindhive/nklab3/users/sahil003/activation_dump_avgpool/noncutout_greyscale_v3
    /mindhive/nklab3/users/sahil003/activation_dump_avgpool/noncutout_greyscale_v4
    /mindhive/nklab3/users/sahil003/activation_dump_avgpool/noncutout_greyscale_v5
    /mindhive/nklab3/users/sahil003/activation_dump_avgpool/noncutout_greyscale_v6
    /mindhive/nklab3/users/sahil003/activation_dump_avgpool/noncutout_greyscale_v7
    /mindhive/nklab3/users/sahil003/activation_dump_avgpool/noncutout_greyscale_v8
    /mindhive/nklab3/users/sahil003/activation_dump_avgpool/noncutout_greyscale_v9
    /mindhive/nklab3/users/sahil003/activation_dump_avgpool/noncutout_greyscale_v10
)

IMAGE_INFO_PATH="floc_image_info.csv"

# Calculate the array index (SLURM_ARRAY_TASK_ID is 1-indexed
INDEX=$(($SLURM_ARRAY_TASK_ID - 1))
ACTIVATION_DIR=${ACTIVATION_DIRS[$INDEX]}

echo "Processing activation directory: $ACTIVATION_DIR on node $(hostname)"

# Run the selectivity analysis on the chosen activation directory.
python selectivity_analysis_parallel.py "$ACTIVATION_DIR" "$IMAGE_INFO_PATH"