import torch
import torch.onnx
from infer_utils import load_diffusionvid

if __name__ == "__main__":
    model_path = "models/DiffusionVID_R101.pth"
    model = load_diffusionvid(model_path)
    
    # Extract the time MLP module
    time_mlp = model.head.time_mlp[0]
    time_mlp.eval()  # Set to evaluation mode
    
    # Load calibration input
    t = torch.load("calib_inputs/sinu_emb_calib_input.pt").cuda()
    
    # Test the module before export
    with torch.no_grad():
        time = time_mlp(t)
    
    print("Input   [:5]:", t.view(-1)[:5])
    print("Output  [:5]:", time.view(-1)[:5])
    print("Output [-5:]:", time.view(-1)[-5:])
    
    # Export to ONNX
    onnx_path = "time_mlp.onnx"
    
    torch.onnx.export(
        time_mlp,                    # model being run
        t,                           # model input (or a tuple for multiple inputs)
        onnx_path,                   # where to save the model
        export_params=True,          # store the trained parameter weights inside the model file
        opset_version=11,            # the ONNX version to export the model to
        do_constant_folding=True,    # whether to execute constant folding for optimization
        input_names=['input'],       # the model's input names
        output_names=['output'],     # the model's output names
        dynamic_axes={               # variable length axes
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to {onnx_path}")
    
    # Optional: Verify the ONNX model
    try:
        import onnx
        import onnxruntime as ort
        
        # Load and check the ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model validation successful!")
        
        # Test ONNX inference
        ort_session = ort.InferenceSession(onnx_path)
        ort_inputs = {ort_session.get_inputs()[0].name: t.cpu().numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)
        
        print("ONNX Output [:5]:", ort_outputs[0].flatten()[:5])
        print("ONNX Output [-5:]:", ort_outputs[0].flatten()[-5:])
        
        # Compare outputs
        torch_output = time.cpu().numpy()
        onnx_output = ort_outputs[0]
        diff = torch.abs(torch.tensor(torch_output) - torch.tensor(onnx_output)).max()
        print(f"Max difference between PyTorch and ONNX: {diff}")
        
    except ImportError:
        print("onnx and/or onnxruntime not installed. Skipping verification.")
    except Exception as e:
        print(f"ONNX verification failed: {e}")
    
    # Save the output for comparison
    torch.save(time, "tmp.pt")