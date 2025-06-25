#!/bin/bash

cd rain-rendering

process_dataset() {
    local date=$1
    local dataset=$2
    local intensity=$3
    local frame_end=$4

    echo "Processing dataset: ${date}/${date}_drive_${dataset}_sync with intensity ${intensity}"


    OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
        python main_threaded.py \
        --dataset_root data/ \
        --dataset KITTI_RAW \
        --intensity $intensity \
        --output data/ \
        --sequence $date/$date"_drive_"$dataset"_sync" \
        --scene_threaded \
        --particles data/particles/ \
        --frame_end $frame_end \
        --frame_start 0 \
        --scenes_per_thread 1
}

# Process each dataset with its specific intensity and frame_end values
echo "Starting rain rendering for all datasets..."

# Intensity 1-9
process_dataset "2011_09_26" "0028" 1 430
process_dataset "2011_09_29" "0026" 1 158
process_dataset "2011_09_26" "0013" 2 144
process_dataset "2011_09_26" "0001" 4 108
process_dataset "2011_09_26" "0064" 5 570
process_dataset "2011_09_26" "0046" 6 125
process_dataset "2011_09_26" "0039" 8 395
process_dataset "2011_09_26" "0017" 9 114
process_dataset "2011_09_26" "0117" 9 660
process_dataset "2011_09_30" "0020" 9 1104

# Intensity 10-30
process_dataset "2011_09_26" "0059" 10 373
process_dataset "2011_09_26" "0015" 15 297
process_dataset "2011_09_26" "0057" 15 361
process_dataset "2011_09_26" "0113" 15 87
process_dataset "2011_09_26" "0029" 25 430
process_dataset "2011_09_30" "0033" 25 1594
process_dataset "2011_09_26" "0009" 30 447
process_dataset "2011_09_30" "0034" 30 1224

# Intensity 31-60
process_dataset "2011_09_26" "0106" 35 227
process_dataset "2011_09_26" "0014" 45 314
process_dataset "2011_09_26" "0011" 50 233
process_dataset "2011_10_03" "0042" 50 1170
process_dataset "2011_09_26" "0104" 55 312
process_dataset "2011_09_30" "0018" 55 2762
process_dataset "2011_10_03" "0034" 60 4663

# Intensity 61-90
process_dataset "2011_09_26" "0061" 65 703
process_dataset "2011_09_26" "0101" 65 936
process_dataset "2011_09_29" "0004" 65 339
process_dataset "2011_09_30" "0016" 65 279
process_dataset "2011_09_26" "0079" 70 100
process_dataset "2011_09_26" "0036" 75 803
process_dataset "2011_09_26" "0096" 75 475
process_dataset "2011_09_30" "0028" 75 5177
process_dataset "2011_09_26" "0022" 85 800
process_dataset "2011_09_26" "0023" 90 474
process_dataset "2011_09_26" "0070" 90 420


# Intensity 91-110
process_dataset "2011_09_26" "0060" 95 78
process_dataset "2011_09_28" "0002" 95 376
process_dataset "2011_09_26" "0020" 100 86
process_dataset "2011_09_26" "0095" 100 268
process_dataset "2011_10_03" "0027" 100 4544
process_dataset "2011_09_26" "0002" 110 77
process_dataset "2011_09_26" "0084" 110 383
process_dataset "2011_09_26" "0093" 110 433

# Intensity 111+
process_dataset "2011_09_26" "0035" 120 131
process_dataset "2011_09_26" "0051" 130 438
process_dataset "2011_09_26" "0052" 140 78
process_dataset "2011_10_03" "0047" 140 837
process_dataset "2011_09_26" "0019" 160 481
process_dataset "2011_09_26" "0091" 160 340
process_dataset "2011_09_26" "0048" 170 22
process_dataset "2011_09_26" "0087" 170 729
process_dataset "2011_09_26" "0027" 180 188
process_dataset "2011_09_26" "0086" 180 706
process_dataset "2011_09_26" "0018" 190 270
process_dataset "2011_09_26" "0005" 200 154
process_dataset "2011_09_28" "0001" 200 106
process_dataset "2011_09_29" "0071" 200 1059
process_dataset "2011_09_30" "0027" 200 1106

# THE ONE THAT WAS MISSING

process_dataset "2011_09_26" "0056" 35 294

echo "All rain rendering completed successfully!"
