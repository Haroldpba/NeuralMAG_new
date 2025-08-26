#!/bin/bash

ulimit -s unlimited
ulimit -v unlimited
ulimit -c unlimited
# Temporarily set PYTHONPATH to include the top-level directory
export PYTHONPATH=$(dirname $(dirname $(pwd))):$PYTHONPATH
# mkdir -p ./log/h200
mkdir -p ./log/halo
for width in 1024;do

    # Define the log file path

    LOG_FILE="./log/halo/speed_w$width.log"

    # Clear the log file if it exists
    > "$LOG_FILE"


    # Run the Python scripts and append output to the log file
    {
        # python unet_speed_base.py --gpu 0 --w $width --layers 2 --trt False --profile False
        # python unet_speed_ori.py --gpu 0 --w $width --layers 2 --trt False --profile False
        # python unet_speed_trt.py --gpu 0 --w $width --layers 2 --trt True
        # python unet_speed_half.py --gpu 0 --w $width --layers 2 --trt True
        # python unet_speed_trt_ori.py --gpu 0 --w $width --layers 2 --trt True
        # python unet_speed_ser.py --gpu 0 --w $width --layers 2 --trt False
        # python unet_speed_pal.py --gpu 0 --w $width --layers 2 --trt False
        python unet_speed_halo.py   --gpu 0 1 2 3 --w $width --layers 2 --trt False --profile False
        # python mm_speed.py   --gpu 0 --w $width --layers 2 --profile False
    } >> "$LOG_FILE"

    # Print the contents of the LOG_FILE
    cat "$LOG_FILE"

    # Extract and print the specific lines from the log file, then append them to the end
    UNet_base_line=$(grep "Unt_base_size:" "$LOG_FILE")
    UNet_ori_line=$(grep "Unt_ori_size:" "$LOG_FILE")
    UNet_trt_line=$(grep "Unt_trt_size:" "$LOG_FILE")
    UNet_half_line=$(grep "Unt_half_size:" "$LOG_FILE")
    UNet_ori_trt_line=$(grep "Unt_ori_trt_size:" "$LOG_FILE")
    Unet_ser_line=$(grep "Unt_ser_size:" "$LOG_FILE")
    UNet_pal_line=$(grep "Unt_pal_size:" "$LOG_FILE")
    UNet_halo_line=$(grep "Unt_halo_size:" "$LOG_FILE")
    MAG_line=$(grep "MAG_size:" "$LOG_FILE")
    echo -e "\n\n +---------------------------Results Summary-----------------------------+"
    echo -e "$UNet_base_line"
    echo -e "$UNet_ori_line"
    echo -e "$UNet_trt_line"
    echo -e "$UNet_half_line"
    echo -e "$UNet_ori_trt_line"
    echo -e "$Unet_ser_line"
    echo -e "$UNet_pal_line"
    echo -e "$UNet_halo_line"
    echo -e "$MAG_line"

    # Append the captured lines to the log file
    {
        echo -e "\n\n +---------------------------Results Summary-----------------------------+"
        echo -e "$UNet_base_line"
        echo -e "$UNet_ori_line"
        echo -e "$UNet_trt_line"
        echo -e "$UNet_half_line"
        echo -e "$UNet_ori_trt_line"
        echo -e "$Unet_ser_line"
        echo -e "$UNet_pal_line"
        echo -e "$UNet_halo_line"
        echo -e "$MAG_line"
    } >> "$LOG_FILE"
done