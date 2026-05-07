import numpy as np

def split_sequence(sequence, n_steps):
    """Split a time series into samples using a sliding window."""
    X, y = [], []
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def mape_calculate(actual, predicted):
    """Compute the MAPE (Mean Absolute Percentage Error)."""
    actual, predicted = np.array(actual), np.array(predicted)
    return np.mean(np.abs((actual - predicted) / actual)) * 100