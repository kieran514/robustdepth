python3 Robust-Depth/train.py \
--model_name Robust-Depth \
--dataset kitti \
--data_path /data/KITTI_RAW \
--eval_split eigen_zhou \
--split eigen_zhou \
--height 192 \
--width 640 \
--cuda 0 \
--weighter 0.01 \
--disparity_smoothness 0.001 \
--batch_size 12 \
--log_frequency 3500 \
--learning_rate 1e-4 \
--num_epochs 30 \
--num_workers 4 \
--val_num_workers 4 \
--scheduler_step_size 20 25 29 \
--weights_init pretrained \
--load_weights_folder None \
--teacher \
--depth_loss \
--warp_clear \
--use_augpose_loss \
--do_gauss --do_shot --do_impulse --do_defocus --do_glass \
--do_zoom --do_snow --do_frost --do_elastic --do_pixelate \
--do_jpeg_comp --do_color --do_blur --do_night --do_rain \
--do_scale --do_tiling --do_vertical --do_erase --do_flip \
--do_greyscale --do_ground_snow --do_dusk --do_dawn --do_fog \
--R --G --B

# --do_fog

