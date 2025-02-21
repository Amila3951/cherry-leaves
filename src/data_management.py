import streamlit as st
import pandas as pd
import base64
import pickle  # Import the pickle library

def download_dataframe_as_csv(df, filename="data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
    return href

def load_pkl_file(file_path):
    """
    Loads a pickle file.

    Args:
        file_path: Path to the pickle file.

    Returns:
        The loaded data.
    """
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data