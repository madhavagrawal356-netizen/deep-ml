import numpy as np

def tile_coding_step(base_alpha: float, n_tilings: int, weights: list, active_tiles: list, target: float) -> dict:
    """
    Perform a single value function update using tile coding with adjusted step size.
    
    Args:
        base_alpha: The base learning rate before adjustment
        n_tilings: Number of tilings used in the tile coding scheme
        weights: List of current weight values for all tiles
        active_tiles: List of indices of tiles active for the current state
        target: The target value (e.g., a return or TD target)
    
    Returns:
        Dictionary with 'adjusted_alpha', 'prediction_before',
        'prediction_after', and 'updated_weights'
    """
    updated_weights = np.array(weights, dtype=float, copy=True)
    active_tiles = np.asarray(active_tiles, dtype=int)

    adjusted_alpha = base_alpha / n_tilings
    prediction_before = float(updated_weights[active_tiles].sum())
    error = target - prediction_before

    np.add.at(updated_weights, active_tiles, adjusted_alpha * error)

    prediction_after = float(updated_weights[active_tiles].sum())

    return {
        "adjusted_alpha": np.round(adjusted_alpha,4),
        "prediction_before": np.round(prediction_before,4),
        "prediction_after": np.round(prediction_after,4),
        "updated_weights": np.round(updated_weights,4).tolist(),
    }
    