"""Finite-difference gradient and Jacobian estimation."""
from __future__ import annotations
import numpy as np


def _validate_method(method):
    if method not in {"central", "forward"}:
        raise ValueError("method must be 'central' or 'forward'")


def grad(f, x, h=1e-5, method="central"):
    
    _validate_method(method)
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h must be a positive finite number")
    x_arr = np.asarray(x, dtype=float)
    scalar = x_arr.ndim == 0
    flat = x_arr.reshape(-1).copy()

    def scalar_eval(point):
        value = np.asarray(f(point.reshape(x_arr.shape) if not scalar else float(point[0])))
        if value.size != 1:
            raise ValueError("f must return a scalar for grad")
        return float(value.reshape(-1)[0])

    base = scalar_eval(flat) if method == "forward" else None
    result = np.empty_like(flat, dtype=float)
    for i in range(flat.size):
        plus = flat.copy(); plus[i] += h
        if method == "central":
            minus = flat.copy(); minus[i] -= h
            result[i] = (scalar_eval(plus) - scalar_eval(minus)) / (2.0 * h)
        else:
            result[i] = (scalar_eval(plus) - base) / h
    if scalar:
        return float(result[0])
    return result.reshape(x_arr.shape)


def jacobian(F, x, h=1e-5, method="central"):
    
    _validate_method(method)
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h must be a positive finite number")
    x_arr = np.asarray(x, dtype=float)
    shape = x_arr.shape
    flat = x_arr.reshape(-1).copy()

    def eval_vector(point):
        arg = point.reshape(shape) if shape else float(point[0])
        out = np.asarray(F(arg), dtype=float)
        if out.size == 0:
            raise ValueError("F must return at least one value")
        return out.reshape(-1)

    base = eval_vector(flat) if method == "forward" else None
    result = None
    for i in range(flat.size):
        plus = flat.copy(); plus[i] += h
        if method == "central":
            minus = flat.copy(); minus[i] -= h
            column = (eval_vector(plus) - eval_vector(minus)) / (2.0 * h)
        else:
            column = (eval_vector(plus) - base) / h
        if result is None:
            result = np.empty((column.size, flat.size), dtype=float)
        elif result.shape[0] != column.size:
            raise ValueError("F must return outputs with a consistent shape for all inputs")
        result[:, i] = column
    if result is None:
        raise ValueError("x must have at least one element")
    return result
