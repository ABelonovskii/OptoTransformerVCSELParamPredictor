import sqlite3
import pandas as pd
from src.constants import THRESHOLD_GAIN


def load_data(namedb, model_type):
    """Loading and primary filtering of data."""
    conn = sqlite3.connect(namedb)
    query = "SELECT input_data, eigenmodes_solution, freq_threshold_solution FROM vcsel_exp_data"
    df = pd.read_sql_query(query, conn)
    conn.close()
    df = df[df['eigenmodes_solution'] != '[]']
    if model_type == THRESHOLD_GAIN:
        df = df[df['freq_threshold_solution'] != '[]']
    return df
