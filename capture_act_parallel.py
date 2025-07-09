import sys
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.encoder_q = models.resnet18()
        self.encoder_q.fc = nn.Sequential(
            nn.Linear(self.encoder_q.fc.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        self.encoder_k = models.resnet18()
        self.encoder_k.fc = nn.Sequential(
            nn.Linear(self.encoder_k.fc.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        self.queue = nn.Parameter(torch.zeros(128, 65536))
        self.queue_ptr = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        q = self.encoder_q(x)
        k = self.encoder_k(x)
        return q, k

def load_custom_model_exact(model_path):
    print(f"Loading checkpoint from: {model_path}")
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model = CustomModel()
    state_dict = checkpoint['state_dict']
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    print("Model successfully loaded with exact dimensions.")
    return model

def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

def register_specific_layer_hooks(model):
    activations = {}

    def hook_fn(layer_name):
        def hook(module, input, output):
            activations[layer_name] = output.detach().cpu().numpy()
        return hook

    hooks = []
    hooks.append(model.encoder_q.conv1.register_forward_hook(hook_fn("conv1")))
    for i, layer_name in enumerate(["layer1", "layer2", "layer3", "layer4"], start=2):
        first_block = getattr(model.encoder_q, layer_name)[0]
        hooks.append(first_block.conv1.register_forward_hook(hook_fn(f"conv{i}")))
    # avg pool hooks!!
    hooks.append(model.encoder_q.avgpool.register_forward_hook(hook_fn("avgpool")))
    hooks.append(model.encoder_q.fc.register_forward_hook(hook_fn("fc6_128")))
    return hooks, activations

def save_activations(activations, image_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for layer_name, activation in activations.items():
        npy_path = os.path.join(output_dir, f"{image_name}_{layer_name}.npy")
        np.save(npy_path, activation)

def capture_activations(model, image_info_path, output_dir, transform):
    hooks, activations = register_specific_layer_hooks(model)
    df = pd.read_csv(image_info_path)
    for _, row in df.iterrows():
        image_path = row['image_path']
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        print(f"Processing image: {image_name}")
        image_tensor = preprocess_image(image_path, transform)
        with torch.no_grad():
            _ = model.encoder_q(image_tensor)
        save_activations(activations, image_name, output_dir)
        activations.clear()
    for hook in hooks:
        hook.remove()

# This is for varying face, varying food models.
# if __name__ == "__main__":
#     model_names = (
#         [f"varying_face_0%_v{i}" for i in range(1,4)] + 
#         [f"varying_face_33%_v{i}" for i in range(1,4)] + 
#         [f"varying_face_100%_v{i}" for i in range(1,4)] + 
#         [f"varying_food_0%_v{i}" for i in range(1,4)] + 
#         [f"varying_food_33%_v{i}" for i in range(1,4)] + 
#         [f"varying_food_100%_v{i}" for i in range(1,4)]
#     )
    
#     # Paths configuration
#     MODEL_PATH = "/mindhive/nklab3/users/bowen/ecoset_models/"
#     IMAGE_INFO_PATH = "floc_image_info.csv"
#     OUTPUT_DIR = "/mindhive/nklab3/users/sahil003/activation_dump/"

#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])

#     # Get model index from SLURM_ARRAY_TASK_ID; array IDs are 1-indexed.
#     try:
#         model_index = int(sys.argv[1]) - 1
#     except (IndexError, ValueError):
#         print("Please provide a valid model index as an argument.")
#         sys.exit(1)

#     if model_index < 0 or model_index >= len(model_names):
#         print(f"Invalid model index: {model_index + 1}")
#         sys.exit(1)

#     model_name = model_names[model_index]
#     print(f"Selected model: {model_name}")

#     # Determine the checkpoint file based on model type
#     if "varying_face" in model_name:
#         if "_0%" in model_name:
#             version = model_name[-1]
#             model_file = f"varying_face/0%/{version}/checkpoint_0099.pth.tar"
#         elif "_33%" in model_name:
#             version = model_name[-1]
#             model_file = f"varying_face/33%/{version}/checkpoint_0099.pth.tar"
#         elif "_100%" in model_name:
#             version = model_name[-1]
#             model_file = f"varying_face/100%/{version}/checkpoint_0099.pth.tar"
#     elif "varying_food" in model_name:
#         if "_0%" in model_name:
#             version = model_name[-1]
#             model_file = f"varying_food/0%/{version}/checkpoint_0099.pth.tar"
#         elif "_33%" in model_name:
#             version = model_name[-1]
#             model_file = f"varying_food/33%/{version}/checkpoint_0099.pth.tar"
#         elif "_100%" in model_name:
#             version = model_name[-1]
#             model_file = f"varying_food/100%/{version}/checkpoint_0099.pth.tar"
#     else:
#         print("Model type not recognized.")
#         sys.exit(1)

#     # Load model and process activations
#     full_model_path = MODEL_PATH + model_file
#     model = load_custom_model_exact(full_model_path)
#     capture_activations(model, IMAGE_INFO_PATH, os.path.join(OUTPUT_DIR, model_name), transform)
#     print("Processing complete.")

if __name__ == "__main__":
    # Build list of 30 models:
    # 10 models for each type: cutout_color_final, mixed_final, noncutout_greyscale.
    # Doing only greyscale cuz i messed up.
    
    model_names = (
        [f"cutout_color_final_v{i}" for i in range(1, 11)] +
        [f"mixed_final_v{i}" for i in range(1, 11)] +
        [f"noncutout_greyscale_v{i}" for i in range(1, 11)]
    )
    
    # Paths configuration
    MODEL_PATH = "/mindhive/nklab3/users/bowen/ecoset_models/"
    IMAGE_INFO_PATH = "floc_image_info.csv"
    OUTPUT_DIR = "/mindhive/nklab3/users/sahil003/activation_dump_avgpool/"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Get model index from SLURM_ARRAY_TASK_ID; array IDs are 1-indexed.
    try:
        model_index = int(sys.argv[1]) - 1
    except (IndexError, ValueError):
        print("Please provide a valid model index as an argument.")
        sys.exit(1)

    if model_index < 0 or model_index >= len(model_names):
        print(f"Invalid model index: {model_index + 1}")
        sys.exit(1)

    model_name = model_names[model_index]
    print(f"Selected model: {model_name}")

    # Determine the checkpoint file based on model type
    if "cutout_color_final" in model_name:
        model_file = "/checkpoint_cutout_color_0099.pth.tar"
    elif "mixed_final" in model_name:
        model_file = "/checkpoint_mixed_0099.pth.tar"
    elif "noncutout_greyscale" in model_name:
        model_file = "/checkpoint_noncutout_greyscale_0099.pth.tar"
    else:
        print("Model type not recognized.")
        sys.exit(1)

    # Load model and process activations
    full_model_path = MODEL_PATH + model_name + model_file
    model = load_custom_model_exact(full_model_path)
    capture_activations(model, IMAGE_INFO_PATH, os.path.join(OUTPUT_DIR, model_name), transform)
    print("Processing complete.")