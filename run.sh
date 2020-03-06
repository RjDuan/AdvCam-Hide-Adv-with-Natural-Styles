#!/bin/bash

start_dir=${1:-`pwd`}

content_dir="$start_dir/physical-attack-data/content/stop-sign/"
content_seg_dir="$start_dir/physical-attack-data/content-mask/"
style_dir="$start_dir/physical-attack-data/style/stop-sign/"
style_seg_dir="$start_dir/physical-attack-data/style-mask/"

for content_img_path in "$content_dir"/*; do
num=(1000)
for i in "${num[@]}"; do
python advcam_main.py  --style_image_path "$style_dir" --style_seg_path "$style_seg_dir" --content_image_path "$content_img_path"  --content_seg_path "$content_seg_dir"  --result_dir './physical_result/stop-signs-lalala/' --attack_weight "$i" --target_label 424
done
done
