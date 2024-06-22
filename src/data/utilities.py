import logging

import torch
from numpy import pi, log10
from src.constants import *

HC9 = 1239.84197596064  # Planck's constant multiplied by the speed of light in nm*eV


TOKEN_RANGES = {
    N: (0.0, 10),  # Refractive index (real part) of the material.
    K: (-1.0, 1.0),  # Extinction coefficient (imaginary part of refractive index) of the material.
    BOUNDARY: None,  # Indicates if the layer is Boundary.
    THICKNESS: (0.0, 1000),  # Thickness of the layer in nanometers.
    GAIN: (0.0, 10000),  # Optical gain of the material expressed in cm^-1.
    ENERGY: (0.0, 10),  # Energy in electronvolts (eV).
    EIGEN_ENERGY_1: (0.0, 10),  # First eigenmode, also in electronvolts (eV).
    EIGEN_ENERGY_2: (0.0, 10),  # Second eigenmode, also in electronvolts (eV).
    Q1: (0.0, 1e5),  # Quality factor of the first eigenmode.
    Q2: (0.0, 1e5),  # Quality factor of the second eigenmode.
    PAIRS_COUNT: (0, 100),  # Number of pairs of layers for DBR in the structure.
    DUAL_LAYER: None,  # Indicates if the layer is dual layer.
    RADIUS: (0, 1000),  # Radius of the first layer in dual-layer configurations, in nanometers.
    END_OF_LAYER: None,  # No range as it's a control token indicating the end of a section (DBR or DUAL_LAYER).
    THRESHOLD_MATERIAL_GAIN: (0.0, 100000)  # Optical gain threshold for the material expressed in cm^-1.
}


def try_parse_float(value):
    try:
        return float(value)
    except ValueError as e:
        logging.error(f"Failed to convert value '{value}' to float: {e}")
        return None


def is_number_in_range(token, number):
    min_val, max_val = TOKEN_RANGES.get(token, (float('-inf'), float('inf')))
    return min_val <= number <= max_val


def normalize_number(token, number):
    min_val, max_val = TOKEN_RANGES[token]
    return (number - min_val) / (max_val - min_val)


def denormalize_number(token, number):
    min_val, max_val = TOKEN_RANGES[token]
    return number * (max_val - min_val) + min_val


def convert_gain_to_k(energy, gain):
    lda0 = HC9 / energy  # Wavelength in nm
    k0 = 2 * pi / lda0  # Vacuum wave number [nm^-1]
    return -gain / k0 / 2 / 10 ** 7  # Refractive index, quantum well, gain domain, imaginary part k0 -> *10^9 gain -> *10^2


def convert_k_to_gain(energy, k):
    lda0 = HC9 / energy
    k0 = 2 * pi / lda0
    return -k * k0 * 2 * 10 ** 7


def log_scale(values, eps=1e-6):
    return log10(max(values, eps))


def log_normalize(values, eps=1e-6, min_val=-6, max_val=6):
    """
    Apply logarithmic transformation followed by normalization to [0, 1].

    Args:
        values: Array of quality factors (or other metrics) to normalize.
        eps: Small constant to avoid log(0).

    Returns:
        Normalized array.
    """
    # Logarithmic transformation
    log_values = log10(max(values, eps))

    # Normalization to [0, 1]
    normalized = (log_values - min_val) / (max_val - min_val)

    return normalized


def log_denormalize(normalized_values, min_val=-6, max_val=6):
    """
    Apply denormalization from [0, 1] to original logarithmic scale, then exponentiate back.

    Args:
        normalized_values: Array of normalized values.
        min_val: Minimum value used in the normalization process.
        max_val: Maximum value used in the normalization process.

    Returns:
        Denormalized array of original values.
    """
    # Denormalization from [0, 1] to the original logarithmic scale
    log_values = normalized_values * (max_val - min_val) + min_val

    # Exponentiate back to original scale
    return 10 ** log_values


def denorm_values(output, targets, model_type):
    if model_type == EIGEN_ENERGY:
        denorm_output = denormalize_number(ENERGY, output)
        denorm_targets = denormalize_number(ENERGY, targets)
    elif model_type == QUALITY_FACTOR:
        denorm_output = log_denormalize(output)
        denorm_targets = log_denormalize(targets)
    elif model_type == THRESHOLD_GAIN:
        denorm_output_energy = denormalize_number('ENERGY', output[:, 0])
        denorm_targets_energy = denormalize_number('ENERGY', targets[:, 0])
        denorm_output_gain = convert_k_to_gain(denorm_output_energy, -output[:, 1])
        denorm_targets_gain = convert_k_to_gain(denorm_targets_energy, -targets[:, 1])
        denorm_output = torch.stack((denorm_output_energy, denorm_output_gain), dim=1)
        denorm_targets = torch.stack((denorm_targets_energy, denorm_targets_gain), dim=1)

    return denorm_output, denorm_targets