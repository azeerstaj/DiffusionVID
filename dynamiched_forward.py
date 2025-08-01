import torch
from infer_utils import load_diffusionvid

if __name__ == "__main__":
    model_path = "models/DiffusionVID_R101.pth"
    model = load_diffusionvid(model_path)
    init_bboxes = torch.randn(1, 300, 4).cuda()  # Example init_bboxes
    features = [torch.randn(1, 256, 72, 128).cuda()]  # Example features
    t = torch.tensor([0]).cuda()  # Example time tensor
    print(model.head(features, init_bboxes, t, None, box_extract=0))
