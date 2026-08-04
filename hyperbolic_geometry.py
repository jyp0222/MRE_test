"""Numerically guarded Lorentz-model operations used by module B.

The public functions in this file use a positive curvature magnitude ``c`` for
a hyperbolic space whose sectional curvature is ``-c``.  A Lorentz point has
shape ``[..., d + 1]`` and follows the convention ``[time, space...]``.
"""

import tensorflow as tf


def _positive_curvature(curvature, dtype):
    curvature = tf.cast(tf.convert_to_tensor(curvature), dtype)
    tf.debugging.assert_positive(
        curvature, message="curvature magnitude must be strictly positive"
    )
    tf.debugging.assert_all_finite(curvature, message="curvature contains NaN or Inf")
    return curvature


def minkowski_inner(x, y, keepdims=False):
    """Return ``-x_time*y_time + <x_space, y_space>``."""
    x = tf.convert_to_tensor(x)
    y = tf.cast(tf.convert_to_tensor(y), x.dtype)
    tf.debugging.assert_equal(
        tf.shape(x), tf.shape(y), message="Minkowski operands must have equal shape"
    )
    result = -x[..., :1] * y[..., :1] + tf.reduce_sum(
        x[..., 1:] * y[..., 1:], axis=-1, keepdims=True
    )
    return result if keepdims else tf.squeeze(result, axis=-1)


def expmap0(tangent, curvature, max_norm=2.0, eps=1e-6):
    """Map Euclidean tangent vectors at the origin to the Lorentz manifold."""
    tangent = tf.convert_to_tensor(tangent)
    if not tangent.dtype.is_floating:
        raise TypeError("tangent must have a floating dtype")
    tf.debugging.assert_rank_at_least(tangent, 2, message="tangent must be [..., d]")
    tf.debugging.assert_all_finite(tangent, message="tangent contains NaN or Inf")
    if max_norm is not None:
        if max_norm <= 0:
            raise ValueError("max_norm must be positive when provided")
        tangent = tf.clip_by_norm(tangent, max_norm, axes=[-1])

    curvature = _positive_curvature(curvature, tangent.dtype)
    sqrt_c = tf.sqrt(curvature)
    radius = tf.norm(tangent, axis=-1, keepdims=True)
    scaled_radius = sqrt_c * radius
    time = tf.cosh(scaled_radius) / sqrt_c
    space_scale = tf.math.divide_no_nan(tf.sinh(scaled_radius), scaled_radius)
    space = space_scale * tangent
    point = tf.concat([time, space], axis=-1)
    return tf.debugging.check_numerics(point, "expmap0 produced NaN or Inf")


def logmap0(point, curvature, eps=1e-6):
    """Map Lorentz points to Euclidean tangent vectors at the origin."""
    point = tf.convert_to_tensor(point)
    if not point.dtype.is_floating:
        raise TypeError("point must have a floating dtype")
    tf.debugging.assert_rank_at_least(point, 2, message="point must be [..., d + 1]")
    tf.debugging.assert_all_finite(point, message="point contains NaN or Inf")

    curvature = _positive_curvature(curvature, point.dtype)
    sqrt_c = tf.sqrt(curvature)
    space = point[..., 1:]
    space_norm = tf.norm(space, axis=-1, keepdims=True)
    acosh_argument = tf.maximum(sqrt_c * point[..., :1], 1.0 + eps)
    distance = tf.math.acosh(acosh_argument) / sqrt_c
    direction = tf.math.divide_no_nan(space, space_norm)
    tangent = distance * direction
    return tf.debugging.check_numerics(tangent, "logmap0 produced NaN or Inf")


def project_curvature(point, source_curvature, target_curvature, max_norm=2.0):
    """Move a point between Lorentz manifolds through their shared origin."""
    point = tf.convert_to_tensor(point)
    source_curvature = _positive_curvature(source_curvature, point.dtype)
    target_curvature = _positive_curvature(target_curvature, point.dtype)
    tangent = logmap0(point, source_curvature)
    tangent *= tf.sqrt(source_curvature / target_curvature)
    return expmap0(tangent, target_curvature, max_norm=max_norm)


def pairwise_distance(query, points, curvature, eps=1e-6):
    """Lorentz distances from ``query [B,d+1]`` to ``points [B,M,d+1]``."""
    query = tf.convert_to_tensor(query)
    points = tf.cast(tf.convert_to_tensor(points), query.dtype)
    tf.debugging.assert_rank(query, 2, message="query must have shape [B, d + 1]")
    tf.debugging.assert_rank(points, 3, message="points must have shape [B, M, d + 1]")
    tf.debugging.assert_equal(
        tf.shape(query)[0], tf.shape(points)[0], message="query/points batch mismatch"
    )
    tf.debugging.assert_equal(
        tf.shape(query)[1], tf.shape(points)[2], message="query/points feature mismatch"
    )
    curvature = _positive_curvature(curvature, query.dtype)
    query = tf.expand_dims(query, axis=1)
    inner = -query[..., :1] * points[..., :1] + tf.reduce_sum(
        query[..., 1:] * points[..., 1:], axis=-1, keepdims=True
    )
    argument = tf.maximum(-curvature * tf.squeeze(inner, axis=-1), 1.0 + eps)
    distance = tf.math.acosh(argument) / tf.sqrt(curvature)
    return tf.debugging.check_numerics(distance, "pairwise_distance produced NaN or Inf")


def normalized_centroid(points, weights, curvature, eps=1e-6):
    """Return a weighted Lorentz centroid normalized back to the manifold."""
    points = tf.convert_to_tensor(points)
    weights = tf.cast(tf.convert_to_tensor(weights), points.dtype)
    tf.debugging.assert_rank(points, 3, message="points must have shape [B, M, d + 1]")
    tf.debugging.assert_rank(weights, 2, message="weights must have shape [B, M]")
    tf.debugging.assert_equal(
        tf.shape(points)[:2], tf.shape(weights), message="points/weights shape mismatch"
    )
    tf.debugging.assert_all_finite(points, message="centroid points contain NaN or Inf")
    tf.debugging.assert_all_finite(weights, message="centroid weights contain NaN or Inf")
    curvature = _positive_curvature(curvature, points.dtype)

    average = tf.reduce_sum(points * tf.expand_dims(weights, axis=-1), axis=1)
    norm_sq = minkowski_inner(average, average, keepdims=True)
    timelike_magnitude = tf.maximum(-norm_sq, eps)
    scale = tf.math.rsqrt(curvature * timelike_magnitude)
    centroid = average * scale
    return tf.debugging.check_numerics(centroid, "normalized_centroid produced NaN or Inf")


def constraint_error(point, curvature):
    """Absolute error of the Lorentz constraint ``<x,x>_L = -1/c``."""
    point = tf.convert_to_tensor(point)
    curvature = _positive_curvature(curvature, point.dtype)
    target = -tf.math.reciprocal(curvature)
    return tf.abs(minkowski_inner(point, point) - target)
