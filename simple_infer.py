from time import time
from infer_utils import *
from mega_core.utils.comm import synchronize

if __name__ == "__main__":

    images_path = "datasets/ILSVRC2015/Data/VID/val/ILSVRC2015_val_00000001"
    model_path = "models/DiffusionVID_R101.pth"
    n_locals = 1
    n_globals = 1
    transform = build_transform()

    total_frames = 10
    cur_frame_id = 0
    start_id = 0
    end_id = total_frames - 1
    frame_category = cur_frame_id != start_id
    last_queue_id = end_id # the latest frame in the queue.

    image_files = sorted(os.listdir(images_path))
    # image_files = (os.listdir(images_path))

    original_image = cv2.imread(
        os.path.join(images_path, image_files[0])
    )

    cur_img = perform_transform(
        transform=transform,
        image = cv2.imread(
            os.path.join(
                images_path, image_files[0]
            )
        )
    )

    ref_l = [
        perform_transform(
            transform=transform, 
            image=cv2.imread(os.path.join(images_path, image_files[i]))
        ) for i in range(n_locals)
    ]

    ref_g = [
        perform_transform(
            transform=transform, 
            image=cv2.imread(os.path.join(images_path, image_files[i]))
        ) for i in range(n_globals)
    ]

    infos = dict(
        cur=cur_img,
        seg_len=total_frames,
        frame_id=cur_frame_id,
        start_id=start_id,
        frame_category=frame_category,
        end_id=end_id,
        last_queue_id=last_queue_id,
        ref_l=ref_l,
        ref_g=ref_g,
    )

    model = load_diffusionvid(model_path)
    
    start_time = time()
    predictions = model(infos)
    # synchronize()  # wait for all processes to finish
    end_time = time()
    
    print(f"FPS: {1/(end_time - start_time)} fps")

    # prediction = predictions[0].to('cpu')
    # prediction = select_top_predictions(prediction)
    # result = draw_image(original_image, cur_img.tensors, prediction)
    # save_image(result)

