def count_parameters(layers: list) -> int:
    """
    Count the total number of trainable parameters in a neural network.
    
    Args:
        layers: A list of dictionaries describing each layer.
                Each dict contains 'type' and layer-specific parameters.
    
    Returns:
        Total number of trainable parameters as an integer.
    """
    # Your code here
    parameters = 0 
    b=0
    for item in layers:
        if item['type']=='dense':
            b = int(item['use_bias'])
            parameters+=item['input_size']*item['output_size']+b*item['output_size']
        elif item['type']=='embedding':
            parameters+=item['num_embeddings']*item['embedding_dim']
        else:
            b = int(item['use_bias'])
            C_in = item['in_channels']
            C_out = item['out_channels']
            if isinstance(item['kernel_size'], int):
                k_h,k_w = item['kernel_size'],item['kernel_size']
            else:
                k_h,k_w = item['kernel_size']
            parameters+=k_h*k_w*C_in*C_out+b*C_out
    return parameters 
