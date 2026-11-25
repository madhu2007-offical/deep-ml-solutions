import numpy as np


def svd_2x2(A: np.ndarray) -> tuple:
    """
    Compute SVD of a 2x2 matrix.
    
    Args:
        A: 2x2 numpy array
    
    Returns:
        U: 2x2 orthogonal matrix (left singular vectors)
        s: 1D array of singular values
        V: 2x2 matrix (right singular vectors, i.e., V^T so A = U @ diag(s) @ V)
    """
    A = np.asarray(A, dtype=float)

    # Form symmetric Gram matrix M = A.T @ A
    M = A.T @ A
    m00, m01, m11 = M[0, 0], M[0, 1], M[1, 1]

    # Exact Jacobi rotation angle to diagonalize M: R^T @ M @ R is diagonal
    theta = 0.5 * np.arctan2(2.0 * m01, m00 - m11)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])

    # Apply right rotation: B = A @ R = U @ diag(singular_values)
    B = A @ R

    # Singular values are the Euclidean norms of the columns of B
    s0 = np.sqrt(np.sum(B[:, 0] ** 2))
    s1 = np.sqrt(np.sum(B[:, 1] ** 2))

    # Sort singular values in descending order and reorder columns of B and R
    if s1 > s0:
        s0, s1 = s1, s0
        B = B[:, [1, 0]]
        R = R[:, [1, 0]]

    singular_values = np.array([s0, s1])

    # Build orthonormal columns of U safely, handling rank-deficient (zero singular value) inputs
    eps = 1e-12
    if s0 > eps:
        u0 = B[:, 0] / s0
    else:
        u0 = np.array([1.0, 0.0])

    if s1 > eps:
        u1 = B[:, 1] / s1
    else:
        # Pick the unit vector orthogonal to u0 so U remains strictly orthogonal even when rank < 2
        u1 = np.array([-u0[1], u0[0]])

    U = np.column_stack([u0, u1])
    V = R.T  # Returned V is V^T so that A = U @ np.diag(singular_values) @ V

    return U, singular_values, V