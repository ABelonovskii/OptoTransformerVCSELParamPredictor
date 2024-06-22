import os
import torch
import logging
import pandas as pd
from torch.utils.data import DataLoader
from datetime import datetime
from src.data.generate_data_embeddings import generate_embeddings
from src.data.utilities import denormalize_number, convert_k_to_gain, log_denormalize
from src.constants import *


def load_prediction_samples(samples_file, model_path):
    """
    Loads and processes prediction samples.

    Args:
        samples_file: Path to the file containing prediction samples.
    """
    with open(samples_file, 'r') as file:
        samples = file.readlines()

    input_lines = [line.strip() for line in samples if line.startswith('2D_AXIAL')]

    df = pd.DataFrame({'input_data': input_lines, 'output_data': [[0.0] for _ in range(len(input_lines))]})
    df = generate_embeddings(df, model_path)

    return df


def predict(model, dataset, args):
    """
    Makes predictions using the provided model and dataset.

    Args:
        model: The trained model instance.
        dataset: Dataset object containing the data for prediction.
        args: Command line arguments containing paths for saving results.
    """
    model.eval()

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    predictions = []

    with torch.no_grad():
        for data in dataloader:
            input_tokens = data['input_tokens']
            output_tokens = model(input_tokens)
            predictions.extend(output_tokens.numpy().tolist())

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    predictions_path = os.path.join(args.predictions_dir, f'predictions_{args.model_type}_{timestamp}.txt')

    os.makedirs(args.predictions_dir, exist_ok=True)

    with open(predictions_path, 'w') as file:
        for prediction in predictions:
            if args.model_type == EIGEN_ENERGY:
                denorm_values = [
                    f"{EIGEN_ENERGY_1} {denormalize_number(ENERGY, prediction[0]):.6f}",
                    f"{EIGEN_ENERGY_2} {denormalize_number(ENERGY, prediction[1]):.6f}"
                ]
            elif args.model_type == QUALITY_FACTOR:
                denorm_values = [
                    f"{Q1} {log_denormalize(prediction[0]):.6f}"
                ]
            elif args.model_type == THRESHOLD_GAIN:
                denorm_values = [
                    f"{THRESHOLD_MATERIAL_GAIN} {convert_k_to_gain(-1 * denormalize_number(ENERGY, prediction[0]), prediction[1]):.6f}",
                    f"{ENERGY} {denormalize_number(ENERGY, prediction[0]):.6f}"
                ]

            file.write(' '.join(denorm_values) + '\n')

    logging.info(f"Predictions saved to {predictions_path}")

