import torch
from mega_core.config import cfg
from mega_core.modeling.detector.diffusion_det import DiffusionDet
from mega_core.modeling.detector.diffusion_det import add_diffusiondet_config
from mega_core.structures.image_list import to_image_list

path = "configs/vid_R_101_DiffusionVID.yaml"
add_diffusiondet_config(cfg)
cfg.merge_from_file(path)

if __name__ == "__main__":

    input_image_shape = [1, 3, 256, 256]
    n_locals = 4
    n_globals = 4

    total_frames = 10
    cur_frame_id = 0
    start_id = 0
    end_id = total_frames - 1
    frame_category = cur_frame_id != start_id
    last_queue_id = end_id # the latest frame in the queue.

    model = DiffusionDet(cfg).cuda()
    model.eval()

    infos = dict(
        cur=torch.randn(input_image_shape).cuda(),
        seg_len=total_frames,
        frame_id=cur_frame_id,
        start_id=start_id,
        frame_category=frame_category,
        end_id=end_id,
        last_queue_id=last_queue_id,
        ref_l=[to_image_list(torch.randn(input_image_shape).cuda()) for _ in range(n_locals)],
        ref_g=[to_image_list(torch.randn(input_image_shape).cuda()) for _ in range(n_globals)],
    )

    predictions = model(infos)

    print("len(predictions):", len(predictions))
    print(predictions[0])