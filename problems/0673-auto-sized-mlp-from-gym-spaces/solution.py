import math
def auto_mlp_from_spaces(obs_space: dict, act_space: dict, hidden_layers: list) -> dict:
    """
    Automatically determine MLP architecture from environment space definitions.

    Args:
        obs_space: dict with 'type' and type-specific keys defining the observation space
        act_space: dict with 'type' and type-specific keys defining the action space
        hidden_layers: list of ints specifying hidden layer widths

    Returns:
        dict with keys:
          'input_dim': int
          'output_dim': int
          'layer_shapes': list of ((weight_rows, weight_cols), (bias_dim,)) tuples
          'total_params': int
    """
    total_params = 0
    if obs_space['type']=='Discrete' or obs_space['type']=='MultiBinary':
        input_dim = obs_space['n']
    else:
        input_dim = math.prod(obs_space['shape'])
    if act_space['type']=='Discrete' or act_space['type']=='MultiBinary':
        output_dim = act_space['n']
    else:
        output_dim = math.prod(act_space['shape'])
    layer_shapes = []
    hidden_layers.insert(0, input_dim)
    hidden_layers.append(output_dim)
    for i in range(len(hidden_layers)-1):
        shape = ((hidden_layers[i], hidden_layers[i+1]), (hidden_layers[i+1],))
        layer_shapes.append(shape)
        total_params+=hidden_layers[i]*hidden_layers[i+1]+hidden_layers[i+1]
    return {'input_dim': input_dim, 'output_dim': output_dim, 'layer_shapes': layer_shapes, 'total_params': total_params}


    