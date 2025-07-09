import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

# Global parameters
N_PER_CLASS = 80
ALPHA = 0.05
LAYERS_TO_ANALYZE = ["conv1", "conv2", "conv3", "conv4", "conv5", "avgpool"]
TARGET_CLASSES = ["face", "scene", "body", "word"]

np.random.seed(0)

def load_activations_and_labels(layer_name, image_info_path, n_per_class, activation_dir):
    df = pd.read_csv(image_info_path)
    # Sample up to n_per_class images per category
    selected_images = []
    for cls in df['category'].unique():
        cls_images = df[df['category'] == cls]
        selected_images += cls_images.sample(n=min(len(cls_images), n_per_class)).to_dict('records')

    activations = []
    labels = []

    for img_info in selected_images:
        image_name = os.path.splitext(os.path.basename(img_info['image_path']))[0]
        cls = img_info['category']
        file_path = os.path.join(activation_dir, f"{image_name}_{layer_name}.npy")
        if not os.path.exists(file_path):
            print(f"Missing file: {file_path}")
            continue
        data = np.load(file_path)
        activations.append(data.flatten())
        labels.append(cls)

    if len(activations) == 0:
        raise ValueError(f"No valid activations found for layer {layer_name}")

    return np.vstack(activations), np.array(labels), np.unique(labels)

def identify_selective_units_one_sided(activations, labels, classes, target_class, alpha):
    other_classes = [cls for cls in classes if cls != target_class]
    pairwise_masks = []

    for other_class in other_classes:
        target_indices = np.where(labels == target_class)[0]
        other_indices = np.where(labels == other_class)[0]
        t_vals, p_vals = ttest_ind(activations[target_indices], activations[other_indices], axis=0, equal_var=False)
        one_sided_p_vals = p_vals / 2
        one_sided_p_vals[t_vals < 0] = 1.0  # Reject only when target activations > other
        reject, _ = multipletests(one_sided_p_vals, alpha=alpha, method="fdr_bh")[:2]
        pairwise_masks.append(reject)

    final_mask = np.all(pairwise_masks, axis=0)
    return np.where(final_mask)[0]

def calculate_fraction_and_save_selective_units(layer_name, activations, labels, classes, alpha, output_dir):
    fractions = {}
    selective_unit_indices = {}

    for cls in TARGET_CLASSES:
        if cls not in classes:
            print(f"Warning: Target class '{cls}' not found in data for layer '{layer_name}'")
            continue

        indices = identify_selective_units_one_sided(activations, labels, classes, cls, alpha)
        fractions[cls] = len(indices) / activations.shape[1]
        selective_unit_indices[cls] = indices

    indices_output_file = os.path.join(output_dir, f"{layer_name}_selective_indices.npy")
    np.save(indices_output_file, selective_unit_indices)

    return fractions

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python selectivity_analysis.py <activation_dir> <image_info_path>")
        sys.exit(1)

    activation_dir = sys.argv[1]
    image_info_path = sys.argv[2]

    print("Processing activation directory:", activation_dir)
    output_dir = os.path.join("final_selective_analysis_output_avgpool", os.path.basename(activation_dir))
    os.makedirs(output_dir, exist_ok=True)

    layer_fractions = []

    for layer_name in LAYERS_TO_ANALYZE:
        print(f"\nProcessing layer: {layer_name}")
        try:
            activations, labels, classes = load_activations_and_labels(layer_name, image_info_path, N_PER_CLASS, activation_dir)
            fractions = calculate_fraction_and_save_selective_units(layer_name, activations, labels, classes, ALPHA, output_dir)
            for cls, fraction in fractions.items():
                layer_fractions.append({"Layer": layer_name, "Class": cls, "Fraction": fraction})
        except Exception as e:
            print(f"Error processing layer {layer_name}: {e}")

    df_fractions = pd.DataFrame(layer_fractions)
    fractions_output_file = os.path.join(output_dir, "fractions_summary.csv")
    df_fractions.to_csv(fractions_output_file, index=False)

    print("\nSummary of selective unit fractions saved.")
    print(df_fractions.to_string(index=False))
