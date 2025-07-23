import cv2
import torch
import numpy as np
from torchvision import transforms as T
from torchvision.transforms import functional as F
from mega_core.modeling.detector import build_detection_model
from mega_core.utils.checkpoint import DetectronCheckpointer
from mega_core.structures.image_list import to_image_list
from mega_core.config import cfg    
from mega_core.modeling.detector.diffusion_det import add_diffusiondet_config

class Resize(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    def get_size(self, image_size):
        w, h = image_size
        size = self.min_size
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        return image

class SimpleInference:
    CATEGORIES = ['_bg_', 'airplane', 'antelope', 'bear', 'bicycle',
                  'bird', 'bus', 'car', 'cattle', 'dog', 'domestic_cat', 
                  'elephant', 'fox', 'giant_panda', 'hamster', 'horse', 
                  'lion', 'lizard', 'monkey', 'motorcycle', 'rabbit',
                  'red_panda', 'sheep', 'snake', 'squirrel', 'tiger', 
                  'train', 'turtle', 'watercraft', 'whale', 'zebra']

    def __init__(self, cfg, confidence_threshold=0.7):
        self.cfg = cfg.clone()
        self.model = build_detection_model(cfg)
        self.model.eval()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)
        
        # Load model weights
        save_dir = cfg.OUTPUT_DIR
        checkpointer = DetectronCheckpointer(cfg, self.model, save_dir=save_dir)
        _ = checkpointer.load(cfg.MODEL.WEIGHT)
        
        self.transforms = self.build_transform()
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        self.cpu_device = torch.device("cpu")
        self.confidence_threshold = confidence_threshold

    def build_transform(self):
        cfg = self.cfg
        
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )
        
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        
        transforms_list = [
            T.ToPILImage(),
            Resize(min_size, max_size),
            T.ToTensor(),
            to_bgr_transform,
        ]
        
        if "diffusion" not in cfg.MODEL.VID.METHOD:
            transforms_list.append(normalize_transform)
            
        return T.Compose(transforms_list)

    def preprocess_image(self, image):
        """Convert OpenCV image to model input format"""
        image_tensor = self.transforms(image)
        image_list = to_image_list(image_tensor, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        return image_list

    def predict(self, image_path, method="base"):
        """
        Run inference on a single image
        
        Args:
            image_path (str): Path to input image
            method (str): Detection method ("base", "mega", "dff", "fgfa", "rdn", "dafa", "diffusion")
            
        Returns:
            predictions: BoxList with detected objects
            annotated_image: Image with bounding boxes and labels
        """
        # Load image
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Preprocess
        image_input = self.preprocess_image(original_image)
        
        # Create infos dictionary based on method
        if method == "base":
            # For base method, just pass the image directly
            with torch.no_grad():
                predictions = self.model(image_input)
        else:
            # For other methods, create infos dictionary
            infos = self.create_infos_dict(image_input, image_path, method)
            print(infos)
            
            # Run inference
            with torch.no_grad():
                predictions = self.model(infos, targets=None)
        
        # Move to CPU
        predictions = [p.to(self.cpu_device) for p in predictions][0]
        
        # Resize predictions to original image size
        height, width = original_image.shape[:-1]
        predictions = predictions.resize((width, height))
        
        # Filter by confidence
        predictions = self.filter_predictions(predictions)
        
        # Create annotated image
        annotated_image = self.draw_predictions(original_image.copy(), predictions)
        
        return predictions, annotated_image

    def filter_predictions(self, predictions):
        """Filter predictions by confidence threshold"""
        scores = predictions.get_field("scores")
        keep = torch.nonzero(scores > self.confidence_threshold).squeeze(1)
        predictions = predictions[keep]
        
        # Sort by score (ascending so higher scores are drawn last)
        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=False)
        return predictions[idx]

    def draw_predictions(self, image, predictions):
        """Draw bounding boxes and labels on image"""
        if len(predictions) == 0:
            return image
            
        # Draw boxes
        image = self.draw_boxes(image, predictions)
        # Draw labels
        image = self.draw_labels(image, predictions)
        
        return image

    def draw_boxes(self, image, predictions):
        """Draw bounding boxes"""
        labels = predictions.get_field("labels")
        colors = self.compute_colors_for_labels(labels).tolist()
        boxes = predictions.bbox

        for box, color in zip(boxes, colors):
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            image = cv2.rectangle(
                image, tuple(top_left), tuple(bottom_right), tuple(color), 2
            )
        return image

    def draw_labels(self, image, predictions):
        """Draw class labels and confidence scores"""
        scores = predictions.get_field("scores").tolist()
        labels = predictions.get_field("labels")
        labels_text = [self.CATEGORIES[i] for i in labels.tolist()]
        boxes = predictions.bbox
        box_colors = self.compute_colors_for_labels(labels).tolist()

        for box, score, label, box_color in zip(boxes, scores, labels_text, box_colors):
            x, y = box[:2]
            text = f"{label}: {score:.2f}"
            
            # Get text size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            
            # Draw background rectangle
            cv2.rectangle(image, (int(x), int(y)), 
                         (int(x + text_size[0]), int(y + text_size[1])), 
                         tuple(box_color), -1)
            
            # Draw text
            cv2.putText(image, text, (int(x), int(y + text_size[1])), 
                       font, font_scale, (255, 255, 255), thickness)
        
        return image

    def create_infos_dict(self, image_input, image_path, method):
        """Create infos dictionary required by the model"""
        infos = {}
        infos["cur"] = image_input
        infos["seg_len"] = 1  # Single image
        infos["frame_id"] = 0
        infos["start_id"] = 0
        infos["frame_category"] = 0  # Key frame
        infos["end_id"] = 0
        infos["last_queue_id"] = 0
        
        # Create PIL transform for reference images
        infos["transforms"] = self.build_pil_transform()
        
        # Set up dummy paths (not used for single image)
        infos["pattern"] = image_path
        infos["img_dir"] = "%s"
        
        if method == "dff":
            infos["is_key_frame"] = True
            
        elif method in ("fgfa", "rdn"):
            # For single image, use the same image as reference
            infos["ref"] = [image_input]
            
        elif method in ("mega", "dafa", "diffusion"):
            # For single image, use the same image as local and global reference
            infos["ref_l"] = [image_input]  # Local references
            infos["ref_g"] = []  # Global references (empty for single image)
            infos["frame_id_g"] = []  # Global frame IDs
            
        return infos

    def build_pil_transform(self):
        """Build PIL transform for reference images"""
        cfg = self.cfg
        
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]] * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x)

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )
        
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        
        transforms_list = [
            Resize(min_size, max_size),
            T.ToTensor(),
            to_bgr_transform,
        ]
        
        if "diffusion" not in cfg.MODEL.VID.METHOD:
            transforms_list.append(normalize_transform)
            
        return T.Compose(transforms_list)
        """Generate colors for different classes"""
        colors = labels[:, None] * self.palette
        colors = (colors % 255).numpy().astype("uint8")
        return colors


if __name__ == "__main__":

    config_file = "configs/vid_R_101_DiffusionVID.yaml"
    model_path = "models/torchvision-R-101.pkl"

    add_diffusiondet_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.VID.MEGA.MIN_OFFSET = -0
    cfg.MODEL.VID.MEGA.MAX_OFFSET = 0
    cfg.MODEL.VID.MEGA.ALL_FRAME_INTERVAL = 1
    cfg.MODEL.VID.MEGA.KEY_FRAME_LOCATION = 0
    cfg.INPUT.INFER_BATCH = 1
    cfg.merge_from_list(["MODEL.WEIGHT", model_path])  # load checkpoint path

    detector = SimpleInference(cfg, confidence_threshold=0.7)

    # Run inference with different methods
    # predictions, annotated_image = detector.predict("test_input_1.JPEG", method="base")
    predictions, annotated_image = detector.predict("test_input_1.JPEG", method="mega")
    
    # Save result
    cv2.imwrite("output.jpg", annotated_image)
    
    # Print detections
    scores = predictions.get_field("scores")
    labels = predictions.get_field("labels")
    for i, (score, label) in enumerate(zip(scores, labels)):
        print(f"Detection {i+1}: {detector.CATEGORIES[label]} ({score:.3f})")