import yaml
import logging
import argparse
from src.utils.logging_config import setup_logging, log_config, log_model_params
from scripts.setup_utils import check_paths, validate_config, validate_model_params, default_model_type
from src.constants import EIGEN_ENERGY, QUALITY_FACTOR, THRESHOLD_GAIN
from src.models.create_model import create_model
from src.data.data_processing import process_data
from src.data.data_loader import load_data
from sklearn.model_selection import train_test_split
from src.data.vcsel_dataset import VCSELDataset
from src.training.train_model import train_model
from src.training.train_model_k_fold import train_model_k_fold
from src.training.test_model import test_model
from src.predict.predict import load_prediction_samples, predict


def main(args):
    setup_logging()
    check_paths(args)
    validate_config(args)
    validate_model_params(args.model_path)

    logging.info("Starting the application")
    log_config(args)
    log_model_params(args.model_path)

    if args.train:
        logging.info("Training the model")
        logging.info("Loading data.")
        data = load_data(args.data_path, args.model_type)

        train_data, evaluate_data = train_test_split(data, test_size=0.1, random_state=42)
        train_data, validate_data = train_test_split(train_data, test_size=0.1, random_state=42)

        logging.info(f"Data split into train ({len(train_data)} records), validate ({len(validate_data)} records), and evaluate ({len(evaluate_data)} records) datasets.")

        train_data = process_data(train_data, args.model_path, args.model_type, is_train=True)
        train_dataset = VCSELDataset(train_data)

        validate_data = process_data(validate_data, args.model_path, args.model_type, is_validate=True)
        validate_dataset = VCSELDataset(validate_data)

        evaluate_data = process_data(evaluate_data, args.model_path, args.model_type, is_test=True)
        evaluate_dataset = VCSELDataset(evaluate_data)

        logging.info(f"Total data {len(train_data)+len(validate_data)+len(evaluate_data)}")
        logging.info(f"Data processing completed successfully.")

        model = create_model(args.model_path, args.load_weights, args.weight_path)
        train_model(model, train_dataset, validate_dataset, evaluate_dataset, args)
        #train_model_k_fold(model, train_dataset, validate_dataset, evaluate_dataset, args)

    if args.test:
        logging.info("Test the model")
        data = load_data(args.data_path, args.model_type)
        data = process_data(data, args.model_path, args.model_type, is_test=True)
        dataset = VCSELDataset(data)

        model = create_model(args.model_path, args.load_weights, args.weight_path)
        test_model(model, dataset, args)

    if args.predict:
        logging.info("Predict")
        samples = load_prediction_samples(args.samples_file, args.model_path)
        model = create_model(args.model_path, args.load_weights, args.weight_path)
        dataset = VCSELDataset(samples)
        predict(model, dataset, args)


if __name__ == '__main__':
    with open("config.yaml", 'r') as file:
        config = yaml.safe_load(file)
    parser = argparse.ArgumentParser(description="OptoTransformer - VCSEL Param Predictor")
    parser.add_argument('--train', action='store_true', default=config['runtime']['train'], help='Train the model')
    parser.add_argument('--test', action='store_true', default=config['runtime']['test'], help='Test the model')
    parser.add_argument('--predict', action='store_true', default=config['runtime']['predict'])
    parser.add_argument('--load_weights', action='store_true', default=config['runtime']['load_weights'])

    parser.add_argument('--data_path', type=str, default=config['data']['path'], help='Path to the data directory')
    parser.add_argument('--model_path', type=str, default=config['model']['params_path'], help='Path to the model')
    parser.add_argument('--weight_path', type=str, default=config['model']['weight_path'], help='Path to the model weights')

    parser.add_argument('--predictions_dir', type=str, default=config['predictions']['directory'], help='Path to the predictions directory')
    parser.add_argument('--samples_file', type=str, default=config['predictions']['samples_file'], help='File with prediction samples')

    parser.add_argument('--batch_size', type=int, default=config['training']['batch_size'])
    parser.add_argument('--num_epochs', type=int, default=config['training']['num_epochs'])
    parser.add_argument('--learning_rate', type=float, default=config['training']['learning_rate'])
    parser.add_argument('--use_scheduler', action='store_true', default=config['training']['use_scheduler'])
    parser.add_argument('--scheduler_factor', type=float, default=config['training']['scheduler_params']['factor'])
    parser.add_argument('--scheduler_patience', type=int, default=config['training']['scheduler_params']['patience'])

    parser.add_argument('--model_type', type=str,  default=default_model_type(config),
                        choices=[EIGEN_ENERGY, QUALITY_FACTOR, THRESHOLD_GAIN],
                        help='Type of model to train or evaluate')

    args = parser.parse_args()
    main(args)
