import yaml
import logging
import torch
from src.models.transformer_model import TransformerModel


def get_model_params(model_path):
    """
    Load model parameters from a YAML configuration file.
    Args:
    model_path (str): Path to the model configuration file.

    Returns:
    dict: Dictionary containing model parameters.
    """
    try:
        with open(model_path, 'r') as file:
            params = yaml.safe_load(file)
            logging.info("Model parameters loaded successfully from {}".format(model_path))
            return params
    except FileNotFoundError:
        logging.error("Configuration file not found at {}".format(model_path))
        raise
    except yaml.YAMLError as e:
        logging.error("Error parsing the configuration file: {}".format(e))
        raise


def create_model(model_path, load_weights, weight_path):
    """
    Create the model instance using parameters loaded from a YAML configuration file.
    Optionally load weights into the model.

    Args:
    model_path (str): Path to the model configuration file.
    load_weights (bool): If True, load weights from the specified path.
    weight_path (str): Path to the weights file.

    Returns:
    ThirdTransformerModel: An instance of the model.
    """
    model_params = get_model_params(model_path)
    model = TransformerModel(**model_params)
    if load_weights:
        model.load_state_dict(torch.load(weight_path))
        logging.info("Weights loaded successfully from {}".format(weight_path))
    return model