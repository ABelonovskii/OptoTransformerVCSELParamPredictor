from src.data.generate_data_embeddings import generate_embeddings
from src.data.normalize_data_entries import process_EQEQ_entries, process_gain_energy_entries
from src.constants import EIGEN_ENERGY, QUALITY_FACTOR, THRESHOLD_GAIN
import pandas as pd
import logging
import torch


def contains_non_empty_lists(list_of_lists):
    return any(len(sublist) > 0 for sublist in list_of_lists)


def process_data(data, model_path, model_type, is_train=False, is_validate=False, is_test=False):
    data_type = 'training' if is_train else 'validation' if is_validate else 'evaluate'
    logging.info(f"Processing {data_type} data.")

    data = generate_embeddings(data, model_path)
    data = data[data['input_data'].apply(contains_non_empty_lists)]


    data['eigenmodes_solution'] = data['eigenmodes_solution'].apply(lambda x: process_EQEQ_entries(x))
    data = data[data['eigenmodes_solution'].apply(len) > 0]

    if model_type == THRESHOLD_GAIN:
        data['freq_threshold_solution'] = data['freq_threshold_solution'].apply(
            lambda x: process_gain_energy_entries(x))
        data = data[data['freq_threshold_solution'].apply(len) > 0]

    data = prapare_data_for_model(data, model_type)
    logging.info(f"Processed {data_type} data: {len(data)} records.")
    return data


def prapare_data_for_model(df, model_type):
    if model_type == EIGEN_ENERGY:
        df = prepare_data_for_eigen_energy(df)
    elif model_type == QUALITY_FACTOR:
        df = prepare_data_for_quality_factor(df)
    elif model_type == THRESHOLD_GAIN:
        df = prepare_data_for_threshold_gain(df)
    return df


def prepare_data_for_eigen_energy(df):
    """
     Prepares data for a model that predicts energies.
     Retrieves only the required columns and generates output_data.
    """
    df['output_data'] = df['eigenmodes_solution'].apply(lambda x: [x[0], x[2] if len(x) > 2 else x[0]])
    return df[['input_data', 'output_data']]


def prepare_data_for_quality_factor(df):
    """
    Prepares data for a model that predicts quality factors.
    Modifies the input data to include energy levels and prepares the output data to be the corresponding quality factors.
    """
    new_rows = []

    for index, row in df.iterrows():
        input_data = row['input_data']
        eigenmodes_and_Q = row['eigenmodes_solution']

        if len(eigenmodes_and_Q) >= 2:
            modified_input_data = [item[:] for item in input_data]
            modified_input_data[0][-1] = eigenmodes_and_Q[0]
            new_rows.append({'input_data': modified_input_data, 'output_data': [eigenmodes_and_Q[1]]})  # Q1

            # uncomment the text below if you want to use the quality factors for the second eigen mode
            #if len(eigenmodes_and_Q) >= 4:
            #    modified_input_data = [item[:] for item in input_data]
            #    modified_input_data[0][-1] = eigenmodes_and_Q[2]
            #    new_rows.append({'input_data': modified_input_data, 'output_data': [eigenmodes_and_Q[3]]})  # Q2

    new_df = pd.DataFrame(new_rows)

    return new_df


def prepare_data_for_threshold_gain(df):
    """
    Prepares data for a model that predicts threshold material gain.
    Modifies the input data to include the first eigen energy and uses the threshold material gain as output data.
    """
    new_rows = []

    for index, row in df.iterrows():
        input_data = row['input_data']
        eigenmodes = row['eigenmodes_solution']
        threshold_gain = row['freq_threshold_solution']

        if len(eigenmodes) > 0:
            modified_input_data = [item[:] for item in input_data]
            modified_input_data[0][-1] = eigenmodes[0]  # Replace the energy level at the first position

            new_rows.append({'input_data': modified_input_data, 'output_data': threshold_gain})

    new_df = pd.DataFrame(new_rows)
    return new_df

