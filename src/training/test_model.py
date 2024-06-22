import logging
import torch
from torch.utils.data import DataLoader
from src.utils.helpers import mape_loss, max_error, r2_score, calculate_sigma
from src.data.utilities import denorm_values
from src.constants import EIGEN_ENERGY, QUALITY_FACTOR, THRESHOLD_GAIN


def test_model(model, test_dataset, args):
    """
    Test the model with the provided test dataset.

    Args:
        model: The trained model instance for test.
        test_dataset: Dataset object containing the testing data.
        args: Command line arguments containing paths и evaluation parameters.
    """

    if args.model_type == THRESHOLD_GAIN:
        return test_threshold_gain_model(model, test_dataset, args)

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    criterion = torch.nn.MSELoss()

    model.eval()

    total_loss = 0
    total_mape_loss = 0
    total_max_error = 0
    all_mape_losses = []


    logging.info("Test")
    count = 0
    with torch.no_grad():
        for data in test_dataloader:
            input_tokens, targets = data['input_tokens'], data['output_tokens']
            outputs = model(input_tokens)
            loss = criterion(outputs, targets)
            denorm_output, denorm_target = denorm_values(outputs, targets, args.model_type)
            mape = mape_loss(denorm_output, denorm_target)
            max_err = max_error(denorm_output, denorm_target)

            total_loss += loss.item()
            total_mape_loss += mape.item()
            total_max_error = max(total_max_error, max_err)
            all_mape_losses.append(mape.item())

            count += 1

    average_loss = total_loss / count
    average_mape_loss = total_mape_loss / count
    sigma = calculate_sigma(all_mape_losses)

    logging.info(f'Average MSE Loss: {average_loss:.16f}, Average MAPE Loss: {average_mape_loss:.3f}, '
                 f'Average Max Error: {total_max_error:.3f}, Sigma (MAPE): {sigma:.3f}')



def test_threshold_gain_model(model, test_dataset, args):
    """
    Test the Threshold Gain model with the provided test dataset.

    Args:
        model: The trained model instance for test.
        test_dataset: Dataset object containing the testing data.
        args: Command line arguments containing paths и evaluation parameters.
    """
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    criterion = torch.nn.MSELoss()

    model.eval()

    total_loss = 0
    total_mape_loss = 0
    total_max_error = 0
    total_r2 = 0
    all_mape_losses = []

    energy_loss = 0
    energy_mape_loss = 0
    energy_max_error = 0
    all_energy_mape_losses = []

    tmg_loss = 0
    tmg_mape_loss = 0
    tmg_max_error = 0
    all_tmg_mape_losses = []

    logging.info("Test (Threshold Gain Model)")
    count = 0
    with torch.no_grad():
        for data in test_dataloader:
            input_tokens, targets = data['input_tokens'], data['output_tokens']
            outputs = model(input_tokens)
            loss = criterion(outputs, targets)
            denorm_output, denorm_target = denorm_values(outputs, targets, args.model_type)
            mape = mape_loss(denorm_output, denorm_target)
            max_err = max_error(denorm_output, denorm_target)
            r2 = r2_score(denorm_output, denorm_target)

            total_loss += loss.item()
            total_mape_loss += mape.item()
            total_max_error = max(total_max_error, max_err)
            total_r2 += r2
            all_mape_losses.append(mape.item())

            energy_loss += criterion(outputs[:, 0], targets[:, 0]).item()
            tmg_loss += criterion(outputs[:, 1], targets[:, 1]).item()

            denorm_output_energy, denorm_target_energy = denorm_output[:, 0], denorm_target[:, 0]
            denorm_output_tmg, denorm_target_tmg = denorm_output[:, 1], denorm_target[:, 1]

            energy_mape_loss += mape_loss(denorm_output_energy, denorm_target_energy).item()
            tmg_mape_loss += mape_loss(denorm_output_tmg, denorm_target_tmg).item()

            energy_max_error = max(energy_max_error, max_error(denorm_output_energy, denorm_target_energy))
            tmg_max_error = max(tmg_max_error, max_error(denorm_output_tmg, denorm_target_tmg))

            all_energy_mape_losses.append(mape_loss(denorm_output_energy, denorm_target_energy).item())
            all_tmg_mape_losses.append(mape_loss(denorm_output_tmg, denorm_target_tmg).item())

            count += 1

    average_loss = total_loss / count
    average_mape_loss = total_mape_loss / count
    average_r2 = total_r2 / count
    sigma = calculate_sigma(all_mape_losses)

    logging.info(f'Average MSE Loss: {average_loss:.16f}, Average MAPE Loss: {average_mape_loss:.3f}, '
                 f'Average Max Error: {total_max_error:.3f}, Average R2 Score: {average_r2:.3f}, '
                 f'Sigma (MAPE): {sigma:.3f}')

    average_energy_loss = energy_loss / count
    average_tmg_loss = tmg_loss / count

    average_energy_mape_loss = energy_mape_loss / count
    average_tmg_mape_loss = tmg_mape_loss / count

    energy_sigma = calculate_sigma(all_energy_mape_losses)
    tmg_sigma = calculate_sigma(all_tmg_mape_losses)

    logging.info(f'Energy - Average MSE Loss: {average_energy_loss:.16f}, Average MAPE Loss: {average_energy_mape_loss:.3f}, '
                 f'Average Max Error: {energy_max_error:.3f}, '
                 f'Sigma (MAPE): {energy_sigma:.3f}')

    logging.info(f'TMG - Average MSE Loss: {average_tmg_loss:.16f}, Average MAPE Loss: {average_tmg_mape_loss:.3f}, '
                 f'Average Max Error: {tmg_max_error:.3f}, '
                 f'Sigma (MAPE): {tmg_sigma:.3f}')