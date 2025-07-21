# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import importlib
import importlib.util
import sys

# Simplified approach - modern PyTorch requires Python 3.7+
def import_file(module_name, file_path, make_importable=False):
    """
    Dynamically import a Python module from a file path.
    
    Args:
        module_name (str): Name to assign to the imported module
        file_path (str): Path to the Python file to import
        make_importable (bool): Whether to add the module to sys.modules
    
    Returns:
        module: The imported module object
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Could not load spec for {module_name} at {file_path}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    if make_importable:
        sys.modules[module_name] = module
    
    return module