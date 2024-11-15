"""
This file will clean demonstrations/tiger_dataset.npy
"""

import os
import numpy as np
import pickle
from io import BytesIO

def load_and_filter_data(file_path, chunk_size=1024 ** 2):
    """
    Load and filter a large NPY or Pickle file containing an array of dictionaries in chunks.

    Parameters:
    file_path (str): Path to the NPY or Pickle file.
    chunk_size (int): Size of each chunk to load in bytes (default is 1MB).

    Returns:
    list of dict: The filtered data.
    """
    file_size = os.path.getsize(file_path)
    num_chunks = (file_size + chunk_size - 1) // chunk_size

    filtered_data = []

    with open(file_path, 'rb') as file:
        for _ in range(num_chunks):
            # Load a chunk of data
            chunk = file.read(chunk_size)

            try:
                data_chunk = _load_pickled_chunk(BytesIO(chunk))
            except EOFError:
                # If the chunk is truncated, we've reached the end of the file
                break

            # Filter the data as needed
            filtered_chunk = [filter_dict(d) for d in data_chunk]
            filtered_data.extend(filtered_chunk)

    return filtered_data

def _load_pickled_chunk(chunk_bytes):
    """
    Load a Pickle chunk, handling potential truncation.
    """
    unpickler = pickle.Unpickler(chunk_bytes)
    data = []
    while True:
        try:
            data.append(unpickler.load())
        except EOFError:
            break
    return data

def filter_dict(data_dict):
    """
    Filter the dictionary data as needed. Modify this function to implement your specific filtering logic.
    """
    # Example: Filter out data where 'foo' is less than 10
    return {k: v for k, v in data_dict.items() if v['foo'] >= 10}

# Example usage
data = load_and_filter_data('demonstrations/tiger_dataset.npy')
