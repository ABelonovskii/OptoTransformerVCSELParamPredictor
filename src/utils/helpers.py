import os
import matplotlib
import matplotlib.pyplot as plt
import torch
from datetime import datetime
import numpy as np


# Use the appropriate backend for matplotlib
matplotlib.use('TkAgg')


def plot_losses(train_losses, test_losses, title):
    """
    Plots both training and testing losses over epochs on the same graph.

    Args:
        train_losses (list): A list of loss values for each training epoch.
        test_losses (list): A list of loss values for each testing epoch.
        title (str): Title for the plot indicating what kind of losses are being plotted.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', marker='o', color='blue')
    if test_losses:
        plt.plot(test_losses, label='Testing Loss', marker='x', color='red')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


def save_losses(weight_path, train_mse_losses, train_mape_losses, train_max_errors, train_r2_scores, test_mse_losses=None, test_mape_losses=None, test_max_errors=None, test_r2_scores=None):
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    losses_filename = f'training_losses_{current_time}.txt'
    losses_path = os.path.join(os.path.dirname(weight_path), losses_filename)
    with open(losses_path, 'w') as f:
        f.write('Epoch\tTrain MSE\tTrain MAPE\tTrain Max Error\tTrain R2\tTest MSE\tTest MAPE\tTest Max Error\tTest R2\n')
        for i in range(len(train_mse_losses)):
            train_mse = train_mse_losses[i]
            train_mape = train_mape_losses[i]
            train_max_err = train_max_errors[i]
            train_r2 = train_r2_scores[i]
            test_mse = test_mse_losses[i] if test_mse_losses else '-'
            test_mape = test_mape_losses[i] if test_mape_losses else '-'
            test_max_err = test_max_errors[i] if test_max_errors else '-'
            test_r2 = test_r2_scores[i] if test_r2_scores else '-'
            f.write(f'{i+1}\t{train_mse}\t{train_mape}\t{train_max_err}\t{train_r2}\t{test_mse}\t{test_mape}\t{test_max_err}\t{test_r2}\n')


def mape_loss(output, target):
    """
    Calculates the Mean Absolute Percentage Error between predictions and true values.
    """
    return torch.mean(torch.abs((target - output) / (target + 1e-8))) * 100


def max_error(output, target):
    return torch.max(torch.abs((target - output) / (target + 1e-8))).item() * 100


def r2_score(output, target):
    if output.shape != target.shape:
        raise ValueError("Output and target must have the same shape.")

    target_mean = torch.mean(target)
    ss_total = torch.sum((target - target_mean) ** 2)

    # Avoid division by zero
    if torch.isclose(ss_total, torch.zeros(1, device=ss_total.device)):
        return torch.tensor(0.0, device=ss_total.device)

    ss_residual = torch.sum((target - output) ** 2)
    return 1 - ss_residual / ss_total


def calculate_sigma(losses):
    return np.std(losses)


def weighted_MSE_loss(inputs, targets):
    weights = targets
    loss = torch.mean(weights * (inputs - targets) ** 2)
    return loss