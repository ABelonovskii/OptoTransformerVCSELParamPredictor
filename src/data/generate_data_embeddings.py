from src.data.utilities import TOKEN_RANGES, try_parse_float, is_number_in_range, normalize_number, convert_gain_to_k
from src.constants import *
import logging
import yaml


def generate_embeddings(df, model_path):
    """
    Applies the create_embeddings function to all elements of the input_data column in the df data frame.
    """
    with open(model_path, 'r') as file:
        model_params = yaml.safe_load(file)
    df['input_data'] = df['input_data'].apply(lambda x: create_embeddings(x, model_params["one_hot_size"]))
    return df


def create_embeddings(s, one_hot_size=11, filter_energy_threshold=0.85):
    """
     Creates a list of embeddings for each token in the string 's'.
     This function is specifically designed for a data processing task,
     where each token in the string is converted into a one-hot embedding.

     The embeddings are formatted as follows:
     [isENERGY, isTHICKNESS, isN, isK, isGAIN, isBOUNDARY, isDBR, isPAIRS_COUNT, isDUAL_LAYER, isRADIUS, value]

     Each flag represents a binary indicator (0 or 1) denoting the presence of the corresponding attribute for the token,
     except for the last element, which contains the normalized numerical value of the token.

     Args:
     s (str): The string containing data to be converted into embeddings.
     one_hot_size (int): The size of the one-hot encoding vector.

     Returns:
     List[List[float]]: A list of embeddings for each token in the original string.
    """
    tokens = s.split()
    embeddings = []

    i = 0
    energy = 0
    isBoundary = 0
    isDBR = 0
    isDUAL_LAYER = 0
    while i < len(tokens):
        token = tokens[i]
        if token in TOKEN_RANGES and i + 1 < len(tokens):

            if token == ENERGY:
                number = try_parse_float(tokens[i + 1])
                energy = number
                if number is None or not is_number_in_range(token, number):
                    return []
                if energy < filter_energy_threshold:
                    return []
                embedding = [0] * one_hot_size
                embedding[0] = 1  # isEnergy
                embedding[-1] = normalize_number(token, number)
                embeddings.append(embedding)
                i += 2

            elif token == BOUNDARY:
                isBoundary = 1
                i += 1

            elif token == THICKNESS:
                number = try_parse_float(tokens[i + 1])
                if number is None or not is_number_in_range(token, number):
                    return []
                embedding = [0] * one_hot_size
                embedding[1] = 1  # isTHICKNESS
                embedding[6] = isDBR  # isDBR
                embedding[8] = isDUAL_LAYER  # isDUAL_LAYER
                embedding[-1] = normalize_number(token, number)
                embeddings.append(embedding)
                i += 2

            elif token == N:
                number = try_parse_float(tokens[i + 1])
                if number is None or not is_number_in_range(token, number):
                    return []
                embedding = [0] * one_hot_size
                embedding[2] = 1  # isN
                embedding[5] = isBoundary  # isBOUNDARY
                embedding[6] = isDBR  # isDBR
                embedding[8] = isDUAL_LAYER  # isDUAL_LAYER
                embedding[-1] = normalize_number(token, number)
                embeddings.append(embedding)
                i += 2

            elif token == K:
                number = try_parse_float(tokens[i + 1])
                if number is None or not is_number_in_range(token, number):
                    return []
                embedding = [0] * one_hot_size
                embedding[3] = 1  # isK
                embedding[5] = isBoundary  # isBOUNDARY
                embedding[6] = isDBR  # isDBR
                embedding[8] = isDUAL_LAYER  # isDUAL_LAYER
                isBoundary = 0
                embedding[-1] = number
                embeddings.append(embedding)
                i += 2

            elif token == GAIN:
                number = try_parse_float(tokens[i + 1])
                if number is None or not is_number_in_range(token, number):
                    return []
                embedding = [0] * one_hot_size
                embedding[4] = 1  # isGAIN
                embedding[6] = isDBR  # isDBR
                embedding[8] = isDUAL_LAYER  # isDUAL_LAYER
                if energy == 0:
                    logging.error(f"The energy is zero.")
                    return []
                embedding[-1] = convert_gain_to_k(energy, number)
                embeddings.append(embedding)
                i += 2

            elif token == PAIRS_COUNT:
                number = try_parse_float(tokens[i + 1])
                if number is None or not is_number_in_range(token, number):
                    return []
                isDBR = 1
                embedding = [0] * one_hot_size
                embedding[6] = isDBR  # isDBR
                embedding[7] = 1  # isPAIRS_COUNT
                embedding[-1] = normalize_number(token, number)
                embeddings.append(embedding)
                i += 2

            elif token == DUAL_LAYER:
                isDUAL_LAYER = 1
                i += 1

            elif token == RADIUS:
                number = try_parse_float(tokens[i + 1])
                if number is None or not is_number_in_range(token, number):
                    return []
                embedding = [0] * one_hot_size
                embedding[8] = isDUAL_LAYER  # isDUAL_LAYER
                embedding[9] = 1  # isRADIUS
                embedding[-1] = normalize_number(token, number)
                embeddings.append(embedding)
                i += 2
            elif token == END_OF_LAYER:
                isDUAL_LAYER = 0
                isBoundary = 0
                isDBR = 0
                i += 1
            else:
                i += 1
        else:
            i += 1

    # additional extension to the general case VCSEL
    embeddings = expand_embeddings(embeddings)

    return embeddings


def expand_embeddings(src, max_size=31):
    """
    A temporary solution to generalize the processing of different kinds of VCSEL structures:
     - Single layer
     - DBR
     - VCSEL
    """
    seq_len, emb_size = len(src), len(src[0])
    expanded_data = [[0] * emb_size for _ in range(max_size)]

    if seq_len == 8:
        fill_indices = {0: 0, 1: 1, 2: 2, 3: 16, 4: 17, 5: 18, 6: 29, 7: 30}
    elif seq_len == 12:
        fill_indices = {i: i for i in range(10)}
        fill_indices.update({10: 29, 11: 30})
    else:
        return src

    for src_idx, target_idx in fill_indices.items():
        expanded_data[target_idx] = src[src_idx]

    return expanded_data


