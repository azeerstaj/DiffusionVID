# python demo/demo.py configs/vid_R_101_C4_1x.yaml \
#     models/diffdet_coco_swinbase.pth \
#     --suffix ".JPEG" \
#     --visualize-path datasets/ILSVRC2015/Data/VID/val/ILSVRC2015_val_00000001 \
#     --output-folder visualization --output-video

# datasets/ILSVRC2015/Data/VID/snippets/val/video_frames

python demo/demo.py configs/vid_R_101_DiffusionVID.yaml \
    models/DiffusionVID_R101.pth \
    --suffix ".JPEG" \
    --visualize-path datasets/ILSVRC2015/Data/VID/val/ILSVRC2015_val_00000001 \
    --output-folder visualization --output-video

# VID_PATH=/home/baykar/git/DiffusionVID/datasets/ILSVRC2015/Data/VID/snippets/val/test_detection.mp4

# python demo/demo.py configs/vid_R_101_DiffusionVID.yaml models/torchvision-R-101.pkl \
#     --video --output-video \
#     --visualize-path $VID_PATH \
#     --output-folder visualization --output-video