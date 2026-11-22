import numpy as np


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
  x_max = np.max(x, axis=axis, keepdims=True)
  exp_x = np.exp(x - x_max)
  return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def _layer_norm(
    x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float
) -> np.ndarray:
  mean = np.mean(x, axis=-1, keepdims=True)
  var = np.var(x, axis=-1, keepdims=True)  # population variance (ddof=0)
  x_norm = (x - mean) / np.sqrt(var + eps)
  return gamma * x_norm + beta


def transformer_encoder_layer(
    X: np.ndarray, weights: dict, num_heads: int, eps: float = 1e-5
) -> np.ndarray:
  batch_size, seq_len, d_model = X.shape
  if d_model % num_heads != 0:
    raise ValueError("d_model must be divisible by num_heads")
  d_k = d_model // num_heads

  # 1. Multi-Head Self-Attention
  Q = X @ weights["W_q"]
  K = X @ weights["W_k"]
  V = X @ weights["W_v"]

  # Reshape and transpose to (batch_size, num_heads, seq_len, d_k)
  Q = Q.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
  K = K.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
  V = V.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)

  # Scaled dot-product attention
  scores = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
  attn_weights = _softmax(scores, axis=-1)
  context = attn_weights @ V

  # Concatenate heads back to (batch_size, seq_len, d_model) and project
  context = context.transpose(0, 2, 1, 3).reshape(
      batch_size, seq_len, d_model
  )
  attn_out = context @ weights["W_o"]

  # 2. First Add & Norm (Post-LN)
  X_norm1 = _layer_norm(X + attn_out, weights["gamma1"], weights["beta1"], eps)

  # 3. Position-wise Feed-Forward Network
  ffn_hidden = np.maximum(0.0, X_norm1 @ weights["W1"] + weights["b1"])
  ffn_out = ffn_hidden @ weights["W2"] + weights["b2"]

  # 4. Second Add & Norm (Post-LN)
  out = _layer_norm(X_norm1 + ffn_out, weights["gamma2"], weights["beta2"], eps)
  return out