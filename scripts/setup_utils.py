import os
import yaml
import logging


def check_paths(args):
    if not os.path.exists(args.data_path):
        logging.error(f"Data path does not exist: {args.data_path}")
        raise FileNotFoundError(f"Data path does not exist: {args.data_path}")
    if not os.path.exists(args.model_path):
        logging.error(f"Model path does not exist: {args.model_path}")
        raise FileNotFoundError(f"Model path does not exist: {args.model_path}")
    if (args.predict or args.load_weights) and not os.path.exists(args.weight_path):
        logging.error(f"Model path does not exist: {args.weight_path}")
        raise FileNotFoundError(f"Model path does not exist: {args.weight_path}")


def validate_config(args):
    if args.batch_size <= 0:
        logging.error("Batch size must be greater than 0.")
        raise ValueError("Batch size must be greater than 0.")
    if args.num_epochs <= 0:
        logging.error("Number of epochs must be greater than 0.")
        raise ValueError("Number of epochs must be greater than 0.")
    if args.learning_rate <= 0:
        logging.error("Learning rate must be positive.")
        raise ValueError("Learning rate must be positive.")
    if (args.predict or args.test) and not args.train and not args.load_weights:
        logging.error("Model weights need to be loaded.")
        raise ValueError("Model weights need to be loaded.")



def validate_model_params(model_path):
    with open(model_path, 'r') as file:
        model_params = yaml.safe_load(file)

    required_keys = ["one_hot_size", "train_emb_size", "nhead", "nhid", "nlayers", "output_size", "dropout",
                     "hidden_sizes"]

    for key in required_keys:
        if key not in model_params:
            logging.error(f"Missing required model parameter: {key}")
            raise ValueError(f"Missing required model parameter: {key}")


def default_model_type(config):
    model_type_ = None
    for model_type, attributes in config['model_types'].items():
        if attributes['enabled']:
            model_type_ = model_type
            break

    if not default_model_type:
        logging.error("No enabled model type is defined in the config.")
        raise ValueError("No enabled model type is defined in the config.")

    return model_type_
