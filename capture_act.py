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
    """
    Register hooks only for specific Conv2d layers and the intermediate 512-dimensional fc layer in encoder_q.
    """
    activations = {}

    def hook_fn(layer_name):
        def hook(module, input, output):
            activations[layer_name] = output.detach().cpu().numpy()
        return hook

    hooks = []

    # Capture conv1
    hooks.append(model.encoder_q.conv1.register_forward_hook(hook_fn("conv1")))

    # Capture first Conv2d in the first BasicBlock of each layer
    for i, layer_name in enumerate(["layer1", "layer2", "layer3", "layer4"], start=2):
        first_block = getattr(model.encoder_q, layer_name)[0]  # First BasicBlock
        hooks.append(first_block.conv1.register_forward_hook(hook_fn(f"conv{i}")))

    # Capture the intermediate 512-dimensional layer in fc
    hooks.append(model.encoder_q.fc[0].register_forward_hook(hook_fn("fc6")))

    # Optionally capture the final fc6 layer (128-dimensional)
    hooks.append(model.encoder_q.fc.register_forward_hook(hook_fn("fc6_128")))

    return hooks, activations

def save_activations(activations, image_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for layer_name, activation in activations.items():
        npy_path = os.path.join(output_dir, f"{image_name}_{layer_name}.npy")
        np.save(npy_path, activation)

def capture_activations(model, image_info_path, output_dir, transform):
    hooks, activations = register_specific_layer_hooks(model)

    # Load image info
    df = pd.read_csv(image_info_path)
    for _, row in df.iterrows():
        image_path = row['image_path']
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        print(f"Processing image: {image_name}")
        image_tensor = preprocess_image(image_path, transform)

        # Forward pass through encoder_q
        with torch.no_grad():
            _ = model.encoder_q(image_tensor)

        # Save activations
        save_activations(activations, image_name, output_dir)
        activations.clear()

    # Remove hooks
    for hook in hooks:
        hook.remove()

# mixed_final_2, noncutout_greyscale_v1 is bad
if __name__ == "__main__":
    model_names = [f"mixed_final_v{i}" for i in range(9,11)]
    MODEL_PATH = "/mindhive/nklab3/users/bowen/ecoset_models/"
    IMAGE_INFO_PATH = "floc_image_info.csv"
    OUTPUT_DIR = "activation_dump/"

    # Preprocessing transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for model_name in model_names:
        if "cutout_color_final" in model_name:
            model_file = "/checkpoint_cutout_color_0099.pth.tar"
        if "mixed_final" in model_name:
            model_file = "/checkpoint_mixed_0099.pth.tar"
        if "noncutout_greyscale" in model_name:
            model_file = "/checkpoint_noncutout_greyscale_0099.pth.tar"

        print("\n--- Loading Model ---")
        model = load_custom_model_exact(MODEL_PATH + model_name + model_file)

        print("\n--- Capturing Activations ---")
        capture_activations(model, IMAGE_INFO_PATH, OUTPUT_DIR + model_name, transform)

        print("\n--- Done ---")
