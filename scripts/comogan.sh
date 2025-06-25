#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e


# Change to the CoMoGAN-main directory
if [ -d "CoMoGAN" ]; then
    cd CoMoGAN
    echo "Changed directory to: $(pwd)"
else
    echo "ERROR: CoMoGAN-main directory not found in the current location."
    echo "Please run this script from the parent directory of CoMoGAN-main."
    exit 1
fi
echo

SCRIPT_CMD="OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CUDA_VISIBLE_DEVICES=8 python scripts/translate.py"

# --- List of unique drive identifiers ---
declare -a DRIVE_IDS=(
    "2011_09_26/2011_09_26_drive_0028_sync"
    "2011_09_26/2011_09_26_drive_0070_sync"
    "2011_09_26/2011_09_26_drive_0001_sync"
    "2011_09_26/2011_09_26_drive_0002_sync"
    "2011_09_26/2011_09_26_drive_0005_sync"
    "2011_09_26/2011_09_26_drive_0009_sync"
    "2011_09_26/2011_09_26_drive_0011_sync"
    "2011_09_26/2011_09_26_drive_0013_sync"
    "2011_09_26/2011_09_26_drive_0014_sync"
    "2011_09_26/2011_09_26_drive_0015_sync"
    "2011_09_26/2011_09_26_drive_0017_sync"
    "2011_09_26/2011_09_26_drive_0018_sync"
    "2011_09_26/2011_09_26_drive_0019_sync"
    "2011_09_26/2011_09_26_drive_0020_sync"
    "2011_09_26/2011_09_26_drive_0022_sync"
    "2011_09_26/2011_09_26_drive_0023_sync"
    "2011_09_26/2011_09_26_drive_0027_sync"
    "2011_09_26/2011_09_26_drive_0029_sync"
    "2011_09_26/2011_09_26_drive_0032_sync"
    "2011_09_26/2011_09_26_drive_0035_sync"
    "2011_09_26/2011_09_26_drive_0036_sync"
    "2011_09_26/2011_09_26_drive_0039_sync"
    "2011_09_26/2011_09_26_drive_0046_sync"
    "2011_09_26/2011_09_26_drive_0048_sync"
    "2011_09_26/2011_09_26_drive_0051_sync" 
    "2011_09_26/2011_09_26_drive_0052_sync"
    "2011_09_26/2011_09_26_drive_0056_sync"
    "2011_09_26/2011_09_26_drive_0057_sync"
    "2011_09_26/2011_09_26_drive_0059_sync"
    "2011_09_26/2011_09_26_drive_0060_sync"
    "2011_09_26/2011_09_26_drive_0061_sync"
    "2011_09_26/2011_09_26_drive_0064_sync"
    "2011_09_26/2011_09_26_drive_0079_sync"
    "2011_09_26/2011_09_26_drive_0084_sync"
    "2011_09_26/2011_09_26_drive_0086_sync"
    "2011_09_26/2011_09_26_drive_0087_sync"
    "2011_09_26/2011_09_26_drive_0091_sync"
    "2011_09_26/2011_09_26_drive_0093_sync"
    "2011_09_26/2011_09_26_drive_0095_sync"
    "2011_09_26/2011_09_26_drive_0096_sync"
    "2011_09_26/2011_09_26_drive_0101_sync"
    "2011_09_26/2011_09_26_drive_0104_sync"
    "2011_09_26/2011_09_26_drive_0106_sync"
    "2011_09_26/2011_09_26_drive_0113_sync"
    "2011_09_26/2011_09_26_drive_0117_sync"
    "2011_09_28/2011_09_28_drive_0001_sync"
    "2011_09_28/2011_09_28_drive_0002_sync"
    "2011_09_29/2011_09_29_drive_0004_sync"
    "2011_09_29/2011_09_29_drive_0026_sync"
    "2011_09_29/2011_09_29_drive_0071_sync"
    "2011_09_30/2011_09_30_drive_0016_sync"
    "2011_09_30/2011_09_30_drive_0018_sync"
    "2011_09_30/2011_09_30_drive_0020_sync"
    "2011_09_30/2011_09_30_drive_0027_sync"
    "2011_09_30/2011_09_30_drive_0028_sync"
    "2011_09_30/2011_09_30_drive_0033_sync"
    "2011_09_30/2011_09_30_drive_0034_sync"
    "2011_10_03/2011_10_03_drive_0027_sync"
    "2011_10_03/2011_10_03_drive_0034_sync"
    "2011_10_03/2011_10_03_drive_0042_sync"
    "2011_10_03/2011_10_03_drive_0047_sync"

)

# Define transformations as an array of strings
# Format: "source_subdir target_subdir phi_value"
declare -a TRANSFORMATIONS=(
    "data     dusk        1.57"
    "data     dawn        4.71"
    "data     night       3.14"
    "rain     dusk+rain   1.57"
    "rain     dawn+rain   4.71"
    "rain     rain+night  3.14"
)

# Loop over each drive identifier
for drive_id in "${DRIVE_IDS[@]}"; do
    echo
    echo "======================================================================="
    echo "Processing Drive ID: $drive_id"
    echo "======================================================================="
    base_path_kitti_raw_data="../../data/KITTI_RAW/$drive_id/image_02"
    
    # Loop over each transformation definition
    for trans_definition in "${TRANSFORMATIONS[@]}"; do
        # Read the space-separated values from the transformation string
        read -r src_subdir target_subdir phi_value <<< "$trans_definition"
        
        # Determine the load path based on source subdirectory
        load_path="$base_path_kitti_raw_data/$src_subdir"
        
        # Save path is always in KITTI_RAW
        save_path="$base_path_kitti_raw_data/$target_subdir/"
        
        echo
        echo "  Executing for:"
        echo "    Load from: $load_path"
        echo "    Save to  : $save_path"
        echo "    Phi      : $phi_value"
        
        # Execute the python script
        "$SCRIPT_CMD" \
            --load_path "$load_path" \
            --save_path "$save_path" \
            --phi "$phi_value"
            
        if [ $? -eq 0 ]; then
            echo "    SUCCESS."
        else
            echo "    ERROR: Command failed for $load_path -> $save_path (phi=$phi_value)"
            # Optionally, you could exit here if one command fails:
            # exit 1
        fi
    done
done

echo
echo "All translations for all configured drives completed."
