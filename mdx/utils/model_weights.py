

# Mahdi Abdollahpour (mahdi.abdollahpour@unibo.it)
# 2025



import h5py
import re


def print_model_layers(model):
    print('printing model layers:\n')
    for i, layer in enumerate(model.layers):
        input_shape = layer.input_shape if hasattr(layer, 'input_shape') else "Unknown"
        output_shape = layer.output_shape if hasattr(layer, 'output_shape') else "Unknown"
        layer_name = layer.name
        layer_type = layer.__class__.__name__

        print(f"{i:02d}: {layer_type} | Name: {layer_name} | Input shape: {input_shape} | Output shape: {output_shape}")


def extract_relevant_name(name, start_token="neural_pusch_receiver/cgnnofdm"):
    """
    Extract the relevant portion of the name starting from a given token.
    """
    start_index = name.find(start_token)
    if start_index != -1:
        return name[start_index:]
    return name
def extract_weights_from_h5_group(group, prefix="", shape=True):
    """
    Recursively extracts weights from an HDF5 group.
    """
    weights = {}
    for key, item in group.items():
        if isinstance(item, h5py.Group):  # If it's a group, recurse into it
            weights.update(extract_weights_from_h5_group(item, prefix=f"{prefix}{key}/",shape=shape))
        elif isinstance(item, h5py.Dataset):  # If it's a dataset, read its shape
            if shape:
                weights[f"{prefix}{key}"] = item.shape
            else:
                weights[f"{prefix}{key}"] = item[()]  # Read dataset as a numpy array
    return weights

def model_comp(model, weights_path, start_token="neural_pusch_receiver/cgnnofdm"):
    model_weights = {
        extract_relevant_name(w.name, start_token): w.shape.as_list() for w in model.trainable_weights
    }

    with h5py.File(weights_path, 'r') as f:
        saved_weights = {
            extract_relevant_name(key, start_token): shape for key, shape in extract_weights_from_h5_group(f).items()
        }

    matching = []
    mismatching = []

    for name, shape in model_weights.items():
        if name in saved_weights:
            if tuple(shape) == saved_weights[name]:
                matching.append((name, shape))
            else:
                mismatching.append((name, shape, saved_weights[name]))
        else:
            mismatching.append((name, shape, "Not in saved weights"))

    for name, shape in saved_weights.items():
        if name not in model_weights:
            mismatching.append((name, "Not in model weights", shape))

    # Print the comparison results
    print("\n=== Matching Weights ===")
    for name, shape in matching:
        print(f"Name: {name} | Shape: {shape}")

    print("\n=== Mismatching or Missing Weights ===")
    for name, model_shape, saved_shape in mismatching:
        print(f"Name: {name} | Model Shape: {model_shape} | Saved Shape: {saved_shape}")

def print_weights(weights, title):
    """
    Prints a formatted list of weights and their shapes.
    """
    print(f"\n=== {title} ===")
    for i, (name, shape) in enumerate(weights.items()):
        print(f"{i:02}: Name: {name} | Shape: {shape}")


def compute_lr_multipliers(model, saved_weights_path, start_token="neural_pusch_receiver/cgnnofdm", lr_m=1):



    model_weights = {
        extract_relevant_name(w.name, start_token): w.shape.as_list() for w in model.trainable_weights
    }
    model_weights = normalize_weights_names(model_weights, prefix="cgnn/readout_ll_rs/")
    model_weights = normalize_weights_names(model_weights, prefix="cgnn/readout_ch_est/")
    with h5py.File(saved_weights_path, 'r') as f:
        saved_weights = {
            extract_relevant_name(key, start_token): shape for key, shape in extract_weights_from_h5_group(f).items()
        }
    saved_weights = normalize_weights_names(saved_weights, prefix="cgnn/readout_ll_rs/")
    saved_weights = normalize_weights_names(saved_weights, prefix="cgnn/readout_ch_est/")


    lr_multipliers = []
    for weight in model.trainable_weights:
        relevant_name = extract_relevant_name(weight.name, start_token)
        if relevant_name in saved_weights:
            if tuple(weight.shape.as_list()) == saved_weights[relevant_name]:
                lr_multipliers.append(lr_m)  # Matching shape
            else:
                lr_multipliers.append(1.0)  # Mismatched shape
        else:
            lr_multipliers.append(1.0)  # Not in saved weights

    return lr_multipliers




import re

def normalize_weights_names(saved_weights, prefix="cgnn/readout_ll_rs/"):
    """
    Normalize the numbers in names after a given prefix, ensuring numbers start from 0,
    sorting and renumbering independently for each unique base_name.

    Parameters:
        saved_weights (dict): A dictionary where keys are names and values are shapes.
        prefix (str): The prefix to target for normalization (e.g., 'cgnn/readout_ll_rs/').

    Returns:
        dict: A dictionary with updated names.
    """
    grouped_names = {}
    updated_weights = {}

    pattern = fr"({prefix}[^/]+)_([\d]+)"

    for name, shape in saved_weights.items():
        match = re.search(pattern, name)
        if match:
            base_name = match.group(1)
            number = int(match.group(2))
            if base_name not in grouped_names:
                grouped_names[base_name] = []
            grouped_names[base_name].append((number, name))  # Store number and original name
        else:

            updated_weights[name] = shape


    for base_name, name_list in grouped_names.items():
        name_list.sort(key=lambda x: x[0])

        num_map = {}
        i = 0
        for new_number, (original_number, original_name) in enumerate(name_list):
            if original_number not in num_map:
                num_map[original_number]=i
                i +=1



            updated_name = original_name.replace(
                f"{base_name}_{original_number}", f"{base_name}_{i}"
            )
            updated_weights[updated_name] = saved_weights[original_name]

    return updated_weights





def transfer_weights_from_h5(model, saved_weights_path, start_token="neural_pusch_receiver/cgnnofdm", verbose=False):
    """
    Load weights from an HDF5 file into a given model. 
    For layers with matching relevant names, weights are set from the file.

    Args:
        model (tf.keras.Model): The model to update weights for.
        saved_weights_path (str): Path to the HDF5 file containing saved weights.
        start_token (str): Token to extract relevant names for matching.

    Returns:
        None
    """

    # Read the weights file
    with h5py.File(saved_weights_path, 'r') as f:
        saved_weights = {
            extract_relevant_name(key, start_token): value for key, value in extract_weights_from_h5_group(f,shape=False).items()
        }

    saved_weights = normalize_weights_names(saved_weights, prefix="cgnn/readout_ll_rs/")
    saved_weights = normalize_weights_names(saved_weights, prefix="cgnn/readout_ch_est/")


    # Get the model's trainable weights and normalize names
    model_weights = {
        # extract_relevant_name(w.name, start_token): w for w in model.trainable_weights
        extract_relevant_name(w.name, start_token): w for w in model.weights
    }

    model_weights = normalize_weights_names(model_weights, prefix="cgnn/readout_ll_rs/")
    model_weights = normalize_weights_names(model_weights, prefix="cgnn/readout_ch_est/")

    matching = []
    mismatching = []

    for name, weight in model_weights.items():
        if name in saved_weights:
            saved_weight = saved_weights[name]
            if tuple(weight.shape.as_list()) == saved_weight.shape:
                try:
                    weight.assign(saved_weight)
                    matching.append((name, weight.shape.as_list()))
                except Exception as e:
                    mismatching.append((name, weight.shape.as_list(), saved_weight.shape, str(e)))
            else:
                mismatching.append((name, weight.shape.as_list(), saved_weight.shape))
        else:
            mismatching.append((name, weight.shape.as_list(), "Not in saved weights"))

    for name, saved_weight in saved_weights.items():
        if name not in model_weights:
            mismatching.append((name, "Not in model weights", saved_weight.shape))


    if verbose:
        # Print the comparison results
        print("\n=== Matching Weights ===")
        for name, shape in matching:
            print(f"Name: {name} | Shape: {shape}")

        print("\n=== Mismatching or Missing Weights ===")
        for entry in mismatching:
            print(" | ".join(map(str, entry)))


        model_weights = {
            extract_relevant_name(w.name, start_token): w for w in model.weights
        }
        model_weights = normalize_weights_names(model_weights, prefix="cgnn/readout_ll_rs/")
        model_weights = normalize_weights_names(model_weights, prefix="cgnn/readout_ch_est/")

        print("\n=== Verifying Updated Weights ===")
        matching_dict = dict(matching)
        for name, weight in model_weights.items():
            if name in saved_weights and name in matching_dict:
                saved_weight = saved_weights[name]
                print(f"Name: {name}")
                print(f"Model Weight Sample: {weight.numpy().flatten()[:5]}")  # Print first 5 elements of the model weight
                print(f"Saved Weight Sample: {saved_weight.flatten()[:5]}")  # Print first 5 elements of the saved weight
