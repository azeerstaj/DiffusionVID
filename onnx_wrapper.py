import torch
import torch.nn as nn
from torch.onnx import export
from mega_core.modeling.detector.detectors import DiffusionDet
from mega_core.config import cfg
import warnings

from mega_core.modeling.detector.diffusion_det import add_diffusiondet_config


class DiffusionDetONNXWrapper(nn.Module):
    """
    ONNX-compatible wrapper for DiffusionDet model.
    This wrapper handles the complex input structure and makes the model ONNX-exportable.
    """
    
    def __init__(self, original_model):
        super().__init__()
        self.model = original_model
        self.model.eval()
        
    def forward(self, cur_images, ref_l_images=None, ref_g_images=None, 
                frame_id=None, start_id=None, end_id=None, seg_len=None, last_queue_id=None):
        """
        Simplified forward pass for ONNX export.
        
        Args:
            cur_images: Current frame images [B, C, H, W]
            ref_l_images: Local reference images [B, C, H, W] (optional)
            ref_g_images: Global reference images [B, C, H, W] (optional)
            frame_id: Current frame ID tensor [B]
            start_id: Start frame ID tensor [B]
            end_id: End frame ID tensor [B]  
            seg_len: Segment length tensor [B]
            last_queue_id: Last queue ID tensor [B]
        """
        # Prepare infos dictionary in the expected format
        infos = {"cur": cur_images}
        
        if ref_l_images is not None:
            infos["ref_l"] = [ref_l_images]  # Wrap in list as expected
            
        if ref_g_images is not None:
            infos["ref_g"] = [ref_g_images]  # Wrap in list as expected
            
        # Add video sequence metadata (use defaults if not provided)
        batch_size = cur_images.shape[0]
        if frame_id is None:
            frame_id = torch.zeros(batch_size, dtype=torch.long)
        if start_id is None:
            start_id = torch.zeros(batch_size, dtype=torch.long)
        if end_id is None:
            end_id = torch.full((batch_size,), 100, dtype=torch.long)  # Default video length
        if seg_len is None:
            seg_len = torch.full((batch_size,), 100, dtype=torch.long)
        if last_queue_id is None:
            last_queue_id = torch.zeros(batch_size, dtype=torch.long)
        
        infos.update({
            "frame_id": frame_id,
            "start_id": start_id, 
            "end_id": end_id,
            "seg_len": seg_len,
            "last_queue_id": last_queue_id,
            "frame_category": 1 if frame_id == 0 else 1 # cant be traced.
        })
        
        # Call the original model's test forward
        return self.model._forward_test(infos["cur"], 
                                       {k: v for k, v in infos.items() if k != "cur"}, 
                                       None)

def convert_diffusion_det_to_onnx(model, output_path, input_shape=(1, 3, 800, 1333)):
    """
    Convert DiffusionDet model to ONNX format.
    
    Args:
        model: The DiffusionDet model instance
        output_path: Path to save the ONNX model
        input_shape: Input image shape (B, C, H, W)
    """
    
    # Set model to evaluation mode
    model.eval()
    
    # Create wrapper
    onnx_model = DiffusionDetONNXWrapper(model)
    
    # Create dummy inputs
    batch_size, channels, height, width = input_shape
    cur_images = torch.randn(batch_size, channels, height, width)
    
    # Optional reference frames (set to None if not using video features)
    ref_l_images = torch.randn(batch_size, channels, height, width)
    ref_g_images = torch.randn(batch_size, channels, height, width)
    
    # Video sequence metadata (dummy values)
    frame_id = torch.tensor([1], dtype=torch.long)  # Current frame ID
    start_id = torch.tensor([0], dtype=torch.long)  # Start of sequence
    end_id = torch.tensor([100], dtype=torch.long)   # End of sequence  
    seg_len = torch.tensor([100], dtype=torch.long)  # Total length
    last_queue_id = torch.tensor([0], dtype=torch.long)  # Last queue frame
    
    # Input names for the ONNX model
    input_names = ['current_images', 'ref_local_images', 'ref_global_images',
                   'frame_id', 'start_id', 'end_id', 'seg_len', 'last_queue_id']
    output_names = ['predictions']
    
    # Dynamic axes for variable input sizes
    dynamic_axes = {
        'current_images': {0: 'batch_size', 2: 'height', 3: 'width'},
        'ref_local_images': {0: 'batch_size', 2: 'height', 3: 'width'},
        'ref_global_images': {0: 'batch_size', 2: 'height', 3: 'width'},
        'frame_id': {0: 'batch_size'},
        'start_id': {0: 'batch_size'},
        'end_id': {0: 'batch_size'},
        'seg_len': {0: 'batch_size'},
        'last_queue_id': {0: 'batch_size'},
        'predictions': {0: 'batch_size'}
    }
    
    try:
        with torch.no_grad():
            # Export to ONNX
            torch.onnx.export(
                onnx_model,
                (cur_images, ref_l_images, ref_g_images, 
                 frame_id, start_id, end_id, seg_len, last_queue_id),
                output_path,
                export_params=True,
                opset_version=11,  # Use opset 11 for better compatibility
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                verbose=True
            )
            print(f"Successfully exported model to {output_path}")
            
    except Exception as e:
        print(f"Error during ONNX export: {e}")
        return False
    
    return True

def simplified_conversion_approach(model, output_path, input_shape=(1, 3, 800, 1333)):
    """
    Alternative approach: Export only the core inference components.
    This bypasses some of the complex data structures.
    """
    
    # Create a simplified wrapper that only handles core inference
    class SimplifiedDiffusionDet(nn.Module):
        def __init__(self, original_model):
            super().__init__()
            self.backbone = original_model.backbone
            self.head = original_model.head
            self.normalizer = original_model.normalizer
            
            # Copy diffusion parameters
            self.num_timesteps = original_model.num_timesteps
            self.sampling_timesteps = original_model.sampling_timesteps
            self.sqrt_alphas_cumprod = original_model.sqrt_alphas_cumprod
            self.sqrt_one_minus_alphas_cumprod = original_model.sqrt_one_minus_alphas_cumprod
            
        def forward(self, x):
            # Normalize input
            x = self.normalizer(x)
            
            # Extract features
            features = self.backbone(x)
            
            # Process through head (this might need further simplification)
            # Note: The head processing might be complex and need custom implementation
            outputs = self.head(features)
            
            return outputs
    
    simplified_model = SimplifiedDiffusionDet(model)
    simplified_model.eval()
    
    dummy_input = torch.randn(input_shape)
    
    try:
        with torch.no_grad():
            torch.onnx.export(
                simplified_model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                            'output': {0: 'batch_size'}},
                verbose=True
            )
            print(f"Successfully exported simplified model to {output_path}")
            return True
            
    except Exception as e:
        print(f"Error during simplified ONNX export: {e}")
        return False

# Usage example
def main():
    config_path = "/home/baykar/git/DiffusionVID/configs/vid_R_101_DiffusionVID.yaml"
    add_diffusiondet_config(cfg)
    cfg.merge_from_file(config_path)
    cfg.MODEL.VID.MEGA.MIN_OFFSET = -0
    cfg.MODEL.VID.MEGA.MAX_OFFSET = 0
    cfg.MODEL.VID.MEGA.ALL_FRAME_INTERVAL = 1
    cfg.MODEL.VID.MEGA.KEY_FRAME_LOCATION = 0
    cfg.INPUT.INFER_BATCH = 1
    
    model = DiffusionDet(cfg)
    # model.load_state_dict(torch.load('your_model.pth'))
    # Method 1: Full model conversion (might face issues)

    success = convert_diffusion_det_to_onnx(model, 'diffusion_det_full.onnx')
    print(f"Full model conversion success: {success}")
    
    # Method 2: Simplified conversion (more likely to succeed)
    # success = simplified_conversion_approach(model, 'diffusion_det_simple.onnx')
    
    pass

# Additional considerations and potential issues:

"""
POTENTIAL ISSUES AND SOLUTIONS:

1. Complex Data Structures:
   - The model expects complex nested dictionaries and lists
   - Solution: Create wrapper that flattens inputs

2. Dynamic Operations:
   - Diffusion sampling involves loops and conditional operations
   - Solution: Pre-compute or simplify the sampling process

3. Custom Operations:
   - Hungarian matching, NMS, and other custom ops may not be ONNX compatible
   - Solution: Replace with ONNX-compatible alternatives or remove for inference

4. Video Processing:
   - The model handles multiple reference frames
   - Solution: Either export each frame separately or batch process

5. Memory Management:
   - The model has complex memory management for video sequences
   - Solution: Simplify for single-frame inference

RECOMMENDED APPROACH:

1. Start with simplified_conversion_approach()
2. If that fails, progressively remove complex components
3. Consider exporting only the backbone + a simplified head
4. For full functionality, you might need to implement custom ONNX operators

TESTING THE EXPORTED MODEL:

import onnxruntime as ort
import numpy as np

# Load and test ONNX model
session = ort.InferenceSession('your_model.onnx')
input_data = np.random.randn(1, 3, 800, 1333).astype(np.float32)
outputs = session.run(None, {'input': input_data})
"""

if __name__ == "__main__":
    main()