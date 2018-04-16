import numpy as np

def softmax(U, rationality):
    # Subtracting away the max prevents overflow.
    U = U - np.max(U)
    exp = np.exp(U/rationality)

    return exp / sum(exp)
