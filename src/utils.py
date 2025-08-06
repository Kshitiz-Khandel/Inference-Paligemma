# Assuming this is pg2_utils.py

from transformers import AutoTokenizer
import json
import glob
from safetensors import safe_open
from typing import Tuple
import os


def load_hf_model(
    model_path: str,
    device: str,
    paligemma_model_class, # Pass the model class dynamically
    paligemma_config_class # Pass the config class dynamically
) -> Tuple[object, AutoTokenizer]: # Use 'object' for a generic type hint
    """
    Loads a PaliGemma model and its tokenizer from a Hugging Face model path.

    Args:
        model_path (str): The path to the Hugging Face model directory.
        device (str): The device to load the model on ('cuda', 'cpu', 'mps').
        paligemma_model_class: The PaliGemmaForConditionalGeneration class (either SWA or Global).
        paligemma_config_class: The PaliGemmaConfig class (either SWA or Global).

    Returns:
        Tuple[object, AutoTokenizer]: A tuple containing the loaded model and tokenizer.
    """
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    assert tokenizer.padding_side == "right"

    # Find all the *.safetensors files
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))

    # ... and load them one by one in the tensors dictionary
    tensors = {}
    for safetensors_file in safetensors_files:
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    # Load the model's config
    with open(os.path.join(model_path, "config.json"), "r") as f:
        model_config_file = json.load(f)
        # Use the passed config class
        config = paligemma_config_class(**model_config_file)

    # Create the model using the configuration and the passed model class
    model = paligemma_model_class(config).to(device)

    # Load the state dict of the model
    model.load_state_dict(tensors, strict=False)

    # Tie weights
    model.tie_weights()

    return (model, tokenizer)






