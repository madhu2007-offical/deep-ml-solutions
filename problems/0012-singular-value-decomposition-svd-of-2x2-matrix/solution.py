import numpy as np


def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    """
    Compute SVD of a 2x2 matrix using one Jacobi rotation.
    
    Args:
        A: A 2x2 numpy array
    
    Returns:
        Tuple (U, S, Vt) where A ≈ U @ diag(S) @ Vt
        - U: 2x2 orthogonal matrix
        - S: length-2 array of singular values
        - Vt: 2x2 orthogonal matrix (transpose of V)
    """
    A = np.asarray(A, dtype=float)

    # Form symmetric Gram matrix M = A.T @ A
    M = A.T @ A
    m00, m01, m11 = M[0, 0], M[0, 1], M[1, 1]

    # Single Jacobi rotation angle theta such that V.T @ M @ V is diagonal
    theta = 0.5 * np.arctan2(2.0 * m01, m00 - m11)
    c, s = np.cos(theta), np.sin(theta)

    # Right singular vector matrix V
    V = np.array([[c, -s], [s, c]])

    # Apply rotation: B = A @ V = U @ diag(S)
    B = A @ V

    # Singular values are the Euclidean norms of the columns of B
    s0 = np.sqrt(np.sum(B[:, 0] ** 2))
    s1 = np.sqrt(np.sum(B[:, 1] ** 2))

    # Sort singular values in descending order and reorder columns of B and V
    if s1 > s0:
        s0, s1 = s1, s0
        B = B[:, [1, 0]]
        V = V[:, [1, 0]]

    S = np.array([s0, s1])

    # Normalize columns of B to get orthogonal matrix U
    U = np.column_stack([B[:, 0] / s0, B[:, 1] / s1])
    Vt = V.T

    return U, S, Vt