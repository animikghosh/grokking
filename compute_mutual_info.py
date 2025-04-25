# compute_mutual_info.py

import numpy as np
import pandas as pd
from experimental_setup import ZivInformationPlane
import ast

def compute_mutual_information(csv_file):
    """
    Computes the mutual information between the input layer and hidden layer,
    and the hidden layer and output layer from a given CSV file.

    Arguments:
    csv_file -- The path to the CSV file containing model weights

    Returns:
    results_dict -- A dictionary with step number and mutual information values
    """

    # Load the CSV file containing both the input data (embeddings) and the hidden layer weights
    weights_df = pd.read_csv(csv_file)

    # Extract the step number from the filename (e.g., model-weights-step-26000.csv -> 26000)
    step_number = int(csv_file.split('-')[-1].split('.')[0])

    # Extract the weights for the 'transformer.embeddings.weight' layer (input data)
    input_data_row = weights_df[weights_df['Layer'] == 'transformer.embeddings.weight']
    input_data_str = input_data_row['Weights'].values[0]  # Get the weights as a string
    input_data = np.array(ast.literal_eval(input_data_str), dtype=float)

    # Extract the weights for the 'transformer.transformer_blocks.0.ff2.weight' layer (hidden layer)
    hidden_layer_weights_row = weights_df[weights_df['Layer'] == 'transformer.transformer_blocks.1.attn.query_proj.weight']
    hidden_layer_weights_str = hidden_layer_weights_row['Weights'].values[0]  # Get the weights as a string
    hidden_layer_weights = np.array(ast.literal_eval(hidden_layer_weights_str), dtype=float)

    # Extract the weights for the 'transformer.output.weight' layer (output layer)
    output_layer_weights_row = weights_df[weights_df['Layer'] == 'transformer.output.weight']
    output_layer_weights_str = output_layer_weights_row['Weights'].values[0]  # Get the weights as a string
    output_layer_weights = np.array(ast.literal_eval(output_layer_weights_str), dtype=float)

    # Ensure the input data and hidden layer weights have the same number of samples
    input_data_length = input_data.shape[0]
    hidden_layer_weights_length = hidden_layer_weights.shape[0]

    # Padding or truncating to ensure both arrays have the same number of samples
    if hidden_layer_weights_length < input_data_length:
        padding_size = input_data_length - hidden_layer_weights_length
        hidden_layer_weights = np.pad(hidden_layer_weights, (0, padding_size), mode='constant', constant_values=0)
    elif hidden_layer_weights_length > input_data_length:
        hidden_layer_weights = hidden_layer_weights[:input_data_length]

    # Ensure the hidden layer weights and output layer weights have the same number of samples
    output_layer_weights_length = output_layer_weights.shape[0]
    if output_layer_weights_length < input_data_length:
        padding_size = input_data_length - output_layer_weights_length
        output_layer_weights = np.pad(output_layer_weights, (0, padding_size), mode='constant', constant_values=0)
    elif output_layer_weights_length > input_data_length:
        output_layer_weights = output_layer_weights[:input_data_length]

    # After processing, ensure they match in size
    assert input_data.shape[0] == hidden_layer_weights.shape[0], "Input and hidden layer weights must have the same number of samples!"
    assert input_data.shape[0] == output_layer_weights.shape[0], "Input and output layer weights must have the same number of samples!"

    # Reshape or flatten the data as needed to ensure compatibility for MI computation
    input_data = input_data.reshape(-1, 1)  # Example reshaping to (N, 1)
    hidden_layer_weights = hidden_layer_weights.reshape(-1, 1)  # Example reshaping to (N, 1)
    output_layer_weights = output_layer_weights.reshape(-1, 1)  # Example reshaping to (N, 1)

    # Compute mutual information for input -> hidden layer and hidden -> output layer
    infoplane = ZivInformationPlane(input_data, hidden_layer_weights)
    IXT, ITW = infoplane.mutual_information(input_data)

    # Compute mutual information for hidden layer -> output layer
    infoplane_hidden_to_output = ZivInformationPlane(hidden_layer_weights, output_layer_weights)
    IHT, IOT = infoplane_hidden_to_output.mutual_information(hidden_layer_weights)

    # Return results as a dictionary
    results_dict = {
        'Step': step_number,
        'MI(X;T) Input to Hidden Layer': IXT,
        'MI(T;Y) Hidden to Output Layer': IHT
    }

    return results_dict
