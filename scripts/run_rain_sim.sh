#!/bin/bash

# Change directory to pytorch-CycleGAN-and-pix2pix
cd pytorch-CycleGAN-and-pix2pix

# Define datasets as arrays
declare -a datasets_2011_09_26=(
    "0001" "0002" "0005" "0009" "0011" "0013" "0014" "0015" "0017" "0018" "0019"
    "0020" "0022" "0023" "0027" "0028" "0029" "0032" "0035" "0036" "0039" "0046"
    "0048" "0051" "0052" "0056" "0057" "0059" "0060" "0061" "0064" "0070" "0079"
    "0084" "0086" "0087" "0091" "0093" "0095" "0096" "0101" "0104" "0106" "0113" "0117"
)

declare -a datasets_2011_09_28=(
    "0001" "0002"
)

declare -a datasets_2011_09_29=(
    "0004" "0026" "0071"
)

declare -a datasets_2011_09_30=(
    "0016" "0018" "0020" "0027" "0028" "0033" "0034"
)

declare -a datasets_2011_10_03=(
    "0027" "0034" "0042" "0047"
)

# Function to process a dataset
process_dataset() {
    local date=$1
    local dataset=$2

    echo "Processing dataset: $date/$date"_drive_"$dataset"_sync

    python test.py \
        --dataroot /home/190229315/Base-Model/data/KITTI_RAW/$date/$date"_drive_"$dataset"_sync"/image_02/data \
        --results_dir /home/190229315/Base-Model/data/KITTI_RAW/$date/$date"_drive_"$dataset"_sync"/image_02/rain_gan/ \
        --name rain_cyclegan \
        --model test \
        --no_dropout \
        --preprocess none \
        --num_test 10000
}

# Process each dataset
echo "Starting processing of 2011_09_26 datasets..."
for dataset in "${datasets_2011_09_26[@]}"; do
    process_dataset "2011_09_26" "$dataset"
done

echo "Starting processing of 2011_09_28 datasets..."
for dataset in "${datasets_2011_09_28[@]}"; do
    process_dataset "2011_09_28" "$dataset"
done

echo "Starting processing of 2011_09_29 datasets..."
for dataset in "${datasets_2011_09_29[@]}"; do
    process_dataset "2011_09_29" "$dataset"
done

echo "Starting processing of 2011_09_30 datasets..."
for dataset in "${datasets_2011_09_30[@]}"; do
        process_dataset "2011_09_30" "$dataset"
done

echo "Starting processing of 2011_10_03 datasets..."
for dataset in "${datasets_2011_10_03[@]}"; do
        process_dataset "2011_10_03" "$dataset"
done


echo "All datasets processed successfully!"
