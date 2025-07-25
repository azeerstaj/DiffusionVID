import torch
from mega_core.structures.image_list import to_image_list
from torchvision import transforms as T
from demo.predictor import Resize
import cv2
from mega_core.config import cfg
from mega_core.modeling.detector.diffusion_det import add_diffusiondet_config
from mega_core.modeling.detector.diffusion_det import DiffusionDet
import os
import PIL


path = "configs/vid_R_101_DiffusionVID.yaml"
add_diffusiondet_config(cfg)
cfg.merge_from_file(path)

CATEGORIES = ['_bg_',  # always index 0
                'airplane', 'antelope', 'bear', 'bicycle',
                'bird', 'bus', 'car', 'cattle',
                'dog', 'domestic_cat', 'elephant', 'fox',
                'giant_panda', 'hamster', 'horse', 'lion',
                'lizard', 'monkey', 'motorcycle', 'rabbit',
                'red_panda', 'sheep', 'snake', 'squirrel',
                'tiger', 'train', 'turtle', 'watercraft',
                'whale', 'zebra']


def save_image(result, filename="debug_annotated_1.jpg", dir = "debug_outputs"):
    os.makedirs(dir, exist_ok=True)
    debug_path = os.path.join(dir, filename)
    PIL.Image.fromarray(result).save(debug_path)
    print(f"Saved annotated image to: {debug_path}")


def load_diffusionvid(ckpt_path):
    checkpoint = torch.load(ckpt_path, weights_only=True)['model']
    new_state_dict = {}
    for k, v in checkpoint.items():
        new_key = k
        if k.startswith("module."):
            new_key = k[len("module."):]
        new_state_dict[new_key] = v

    model = DiffusionDet(cfg).cuda()
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

def draw_image(original_image, cur_img_tensors, prediction):
    height, width = cur_img_tensors.shape[-2:]
    prediction = prediction.resize((width, height))
    result = original_image.copy()
    result = overlay_boxes(result, prediction)
    result = overlay_class_names(result, prediction) # image with boxes and class names
    return result


def compute_colors_for_labels(labels):
    """
    Simple function that adds fixed colors depending on the class
    """
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = labels[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")
    return colors


def overlay_class_names(image, predictions):
    """
    Adds detected class names and scores in the positions defined by the
    top-left corner of the predicted bounding box
    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `scores` and `labels`.
    """
    scores = predictions.get_field("scores").tolist()
    labels = predictions.get_field("labels")
    labels_text = [CATEGORIES[i] for i in labels.tolist()]
    boxes = predictions.bbox
    box_colors = compute_colors_for_labels(labels).tolist()

    template = "{}: {:.2f}"
    for box, score, label, box_color in zip(boxes, scores, labels_text, box_colors):
        x, y = box[:2]
        s = template.format(label, score)
        # self.draw_text(
        #     image, s, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, tuple(box_color)
        # )

    return image

def overlay_boxes(image, predictions):
    try:
        labels = predictions.get_field("labels")
        colors = compute_colors_for_labels(labels).tolist()
    except:
        colors = [[255, 0, 0] for i in range(len(predictions.bbox))] #bgr
    boxes = predictions.bbox

    for box, color in zip(boxes, colors):
        box = box.to(torch.int64)
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        image = cv2.rectangle(
            image, tuple(top_left), tuple(bottom_right), tuple(color), 2
        )

    return image

def select_top_predictions(predictions, threshold=0.5):
    scores = predictions.get_field("scores")
    keep = torch.nonzero(scores > threshold).squeeze(1)
    predictions = predictions[keep]
    scores = predictions.get_field("scores")
    _, idx = scores.sort(0, descending=False)   # ascending order in order that higher score boxes drawn later
    return predictions[idx]

def perform_transform(transform, image, device='cuda'):
    image = transform(image)
    image_list = to_image_list(image, cfg.DATALOADER.SIZE_DIVISIBILITY)
    image_list = image_list.to(device)

    return image_list

def build_transform():
    """
    Creates a basic transformation that was used to train the models
    """

    if cfg.INPUT.TO_BGR255:
        to_bgr_transform = T.Lambda(lambda x: x * 255)
    else:
        to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

    min_size = cfg.INPUT.MIN_SIZE_TEST
    max_size = cfg.INPUT.MAX_SIZE_TEST
    transforms_list = [
        T.ToPILImage(),
        Resize(min_size, max_size),
        T.ToTensor(),
        to_bgr_transform,
    ]
    transform = T.Compose(transforms_list)
    return transform