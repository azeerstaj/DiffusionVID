from torch.fx import symbolic_trace


import torch
from infer_utils import load_diffusionvid

if __name__ == "__main__":
    model_path = "models/DiffusionVID_R101.pth"
    model = load_diffusionvid(model_path)

    graph = symbolic_trace(model.head)
    graph.graph.print_tabular()
    exit(0)

    batch_size = 1
    dim = 16
    t = torch.load("calib_inputs/sinu_emb_calib_input.pt").cuda()

    time = model.head.time_mlp[0](t)
    torch.save(time, "tmp.pt")

    print("Input   [:5]:", t.view(-1)[:5])
    print("Output  [:5]:", time.view(-1)[:5])
    print("Output [-5:]:", time.view(-1)[-5:])