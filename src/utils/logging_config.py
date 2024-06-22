import yaml
import logging


def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename='app.log',
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def log_config(args):
    logging.info("Configuration:")
    logging.info(f"Model type: {args.model_type}")
    logging.info(f"Train: {args.train}")
    logging.info(f"Test: {args.test}")
    logging.info(f"Predict: {args.predict}")
    logging.info(f"Load Weights: {args.load_weights}")
    logging.info(f"Data Path: {args.data_path}")
    logging.info(f"Model Path: {args.model_path}")
    logging.info(f"Weight Path: {args.weight_path}")
    logging.info(f"Batch Size: {args.batch_size}")
    logging.info(f"Number of Epochs: {args.num_epochs}")
    logging.info(f"Learning Rate: {args.learning_rate}")


def log_model_params(model_path):
    with open(model_path, 'r') as file:
        model_params = yaml.safe_load(file)
    logging.info("Model parameters:")
    for key, value in model_params.items():
        logging.info(f"  {key}: {value}")