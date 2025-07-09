import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

ACTIVATION_DIR = "activation"
N_PER_CLASS = 80
ALPHA = 0.05
LAYERS_TO_ANALYZE = ["conv1", "conv2", "conv3", "conv4", "conv5", "fc6"]

TARGET_CLASSES = [f"class{i+1}" for i in range(4)]  # Analyze permuted class labels
 # Only analyze selectivity for these classes


# MAYBE I SHOULD NOT PERMUTE SCRAMBLED??????????

np.random.seed(0)

def load_activations_and_labels(layer_name, image_info_path, n_per_class, n_classes=4):
    df = pd.read_csv(image_info_path)
    selected_images = []

    # Fixed random seed to ensure consistent sampling
    np.random.seed(0)
    
    for cls in df['category'].unique():
        cls_images = df[df['category'] == cls]
        selected_images += cls_images.sample(n=min(len(cls_images), n_per_class), random_state=0).to_dict('records')

    activations = []
    labels = []

    for img_info in selected_images:
        image_name = os.path.splitext(os.path.basename(img_info['image_path']))[0]
        file_path = os.path.join(ACTIVATION_DIR, f"{image_name}_{layer_name}.npy")
        if not os.path.exists(file_path):
            print(f"Missing file: {file_path}")
            continue
        data = np.load(file_path)
        activations.append(data.flatten())
        labels.append(img_info['category'])  # Preserve original labels

    if len(activations) == 0:
        raise ValueError(f"No valid activations found for layer {layer_name}")

    # Fix order, then permute labels
    np.random.seed(0)  # Consistent permutation
    permuted_indices = np.random.permutation(len(labels))
    permuted_labels = [f"class{(i % n_classes) + 1}" for i in permuted_indices]

    return np.vstack(activations), np.array(permuted_labels), [f"class{i+1}" for i in range(n_classes)]


def identify_selective_units_one_sided(activations, labels, classes, target_class, alpha):
    """
    Identify selective neurons for `target_class` using one-sided t-tests.
    """
    other_classes = [cls for cls in classes if cls != target_class]
    n_neurons = activations.shape[1]
    pairwise_masks = []

    for other_class in other_classes:
        target_indices = np.where(labels == target_class)[0]
        other_indices = np.where(labels == other_class)[0]

        # Perform one-sided t-test: Alternative hypothesis -> target > other
        t_vals, p_vals = ttest_ind(
            activations[target_indices], activations[other_indices], axis=0, equal_var=False
        )
        one_sided_p_vals = p_vals / 2  # Convert two-sided p-values to one-sided
        one_sided_p_vals[t_vals < 0] = 1.0  # Adjust for direction (reject only for positive differences)

        reject, _ = multipletests(one_sided_p_vals, alpha=alpha, method="fdr_bh")[:2]
        pairwise_masks.append(reject)

    # Logical AND across all pairwise masks to enforce strict selectivity
    final_mask = np.all(pairwise_masks, axis=0)
    return np.where(final_mask)[0]  # Indices of selective neurons

def calculate_fraction_and_save_selective_units(layer_name, activations, labels, classes, alpha, output_dir):
    fractions = {}
    selective_unit_indices = {}

    # Only analyze selectivity for TARGET_CLASSES
    for cls in TARGET_CLASSES:
        if cls not in classes:
            print(f"Warning: Target class '{cls}' not found in data for layer '{layer_name}'")
            continue

        indices = identify_selective_units_one_sided(activations, labels, classes, cls, alpha)
        fractions[cls] = len(indices) / activations.shape[1]
        selective_unit_indices[cls] = indices

    # Save selective unit indices for each category
    indices_output_file = os.path.join(output_dir, f"{layer_name}_selective_indices.npy")
    np.save(indices_output_file, selective_unit_indices)

    return fractions

if __name__ == "__main__":
    image_info_path = "floc_image_info.csv"
    output_dir = "selective_analysis_output_permuted"
    os.makedirs(output_dir, exist_ok=True)

    layer_fractions = []

    for layer_name in LAYERS_TO_ANALYZE:
        print(f"\nProcessing layer: {layer_name}")
        try:
            activations, labels, classes = load_activations_and_labels(layer_name, image_info_path, N_PER_CLASS)
            fractions = calculate_fraction_and_save_selective_units(layer_name, activations, labels, classes, ALPHA, output_dir)

            for cls, fraction in fractions.items():
                layer_fractions.append({"Layer": layer_name, "Class": cls, "Fraction": fraction})

        except Exception as e:
            print(f"Error processing layer {layer_name}: {e}")

    # Create and display the fraction summary
    df_fractions = pd.DataFrame(layer_fractions)
    df_fractions["Layer_Sort"] = df_fractions["Layer"].str.extract(r'(\d+)', expand=False).astype(int)
    df_fractions = df_fractions.sort_values(by=["Layer_Sort", "Class"]).drop(columns=["Layer_Sort"])

    fractions_output_file = os.path.join(output_dir, "fractions_summary.csv")
    df_fractions.to_csv(fractions_output_file, index=False)

    print("\nSummary of selective unit fractions saved.")
    print(df_fractions.to_string(index=False))
