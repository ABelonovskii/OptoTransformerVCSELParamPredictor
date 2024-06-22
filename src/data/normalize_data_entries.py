from src.data.utilities import try_parse_float, is_number_in_range, normalize_number, convert_gain_to_k, log_normalize
from src.constants import *
import logging


def process_EQEQ_entries(data_string, filter_energy_threshold=0.85, filter_quality_threshold=0):
    """
    Processes the input string containing sequential token-value pairs to extract, normalize, and filter out data
    based on predefined criteria. The function handles pairs designated as energy ("E") followed by a quality ("Q")
    token. Both components of the pair must meet specific range criteria to be included in the output.

    :param data_string: A string containing sequences of tokens and their corresponding numeric values. Expected tokens are 'EIGEN_ENERGY_1', 'EIGEN_ENERGY_2', 'Q1', 'Q2'.
                        Example: "EIGEN_ENERGY_1 0.9 Q1 0.05 EIGEN_ENERGY_2 0.85 Q2 0.08"
    :param filter_energy_threshold: Minimum threshold for energy values to be included in the results.

    :return: A list of normalized values for each valid E-Q pair in the input string. Values are normalized and
             included in the output only if both elements of the pair (E and Q) fall within specified ranges.
    """

    tokens = data_string.split()
    normalized_values = []
    i = 0

    while i < len(tokens) - 3:
        e_token = tokens[i]
        e_value = try_parse_float(tokens[i + 1])
        q_token = tokens[i + 2]
        q_value = try_parse_float(tokens[i + 3])

        if e_token in {EIGEN_ENERGY_1, EIGEN_ENERGY_2} and q_token in {Q1, Q2}:
            if e_value is not None and q_value is not None:
                if is_number_in_range(e_token, e_value) and is_number_in_range(q_token, q_value) and e_value >= filter_energy_threshold and q_value >= filter_quality_threshold:
                    normalized_e = normalize_number(e_token, e_value)
                    normalized_q = log_normalize(q_value)
                    normalized_values.extend([normalized_e, normalized_q])
            i += 4
        else:
            i += 1

    return normalized_values


def process_gain_energy_entries(data_string, filter_energy_threshold=0.85, filter_TMG_threshold=1e6):
    """
    Processes a specific format of input string that must include 'ENERGY', 'TRESHOLD_MATERIAL_GAIN', and optionally 'kappa_QW'.
    The function extracts values for ENERGY and TRESHOLD_MATERIAL_GAIN, checks their ranges, applies a conversion
    function to compute k, and checks k's range. If all checks are passed, it appends the [energy, k] pair to the output.

    :param filter_energy_threshold:
    :param data_string: A string formatted with tokens and values, e.g., "TRESHOLD_MATERIAL_GAIN 101311.828623 ENERGY 0.005782 kappa_QW 0.999578"

    :return: A list containing a single pair [energy, k] if all conditions are met, otherwise an empty list.
    """
    tokens = data_string.split()
    result = []

    try:
        energy_index = tokens.index(ENERGY) + 1
        gain_index = tokens.index(THRESHOLD_MATERIAL_GAIN) + 1
        energy = try_parse_float(tokens[energy_index])
        gain = try_parse_float(tokens[gain_index])

        if is_number_in_range(ENERGY, energy) and is_number_in_range(THRESHOLD_MATERIAL_GAIN, gain) and energy >= filter_energy_threshold:
            if gain <= filter_TMG_threshold:
                k = convert_gain_to_k(energy, gain)
                if is_number_in_range(K, k):
                    result.append(normalize_number(ENERGY, energy))
                    result.append(-1 * k)  # convert k form interval [-1, 0] to interval [0, 1] for the convenience of the neural network
    except (ValueError, IndexError) as e:
        logging.error(f"An error occurred while processing the input: {e}")

    return result
