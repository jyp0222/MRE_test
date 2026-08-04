"""TensorFlow implementation of the adapted curvature-aware module B."""

import math

import tensorflow as tf

from hyperbolic_geometry import (
    constraint_error,
    expmap0,
    logmap0,
    normalized_centroid,
    pairwise_distance,
    project_curvature,
)


def _inverse_sigmoid(value):
    value = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


def _inverse_softplus(value):
    value = max(float(value), 1e-6)
    if value > 20.0:
        return value
    return math.log(math.expm1(value))


class CurvatureAwareFusion(tf.keras.layers.Layer):
    """Fuse text and image features in a learnable shared Lorentz space.

    This is a two-modality adaptation of paper B.  It keeps DAEO's required
    ``[B, 3072]`` interface and learns through DAEO's existing losses; no new
    task loss is introduced.
    """

    def __init__(
        self,
        text_dim=2304,
        image_dim=768,
        output_dim=3072,
        fusion_dim=128,
        residual_scale=0.05,
        curvature_init=1.0,
        curvature_min=0.05,
        curvature_max=5.0,
        prior_strength_init=0.1,
        temperature_init=None,
        tangent_scale=1.0,
        tangent_norm_max=2.0,
        initializer_seed=10000,
        name="module_b_curvature_fusion",
        **kwargs,
    ):
        super(CurvatureAwareFusion, self).__init__(name=name, **kwargs)
        if min(text_dim, image_dim, output_dim, fusion_dim) <= 0:
            raise ValueError("all feature dimensions must be positive")
        if residual_scale < 0:
            raise ValueError("residual_scale must be non-negative")
        if not 0 < curvature_min < curvature_max:
            raise ValueError("curvature bounds must satisfy 0 < min < max")
        if not curvature_min <= curvature_init <= curvature_max:
            raise ValueError("curvature_init must lie inside the configured bounds")
        if prior_strength_init < 0:
            raise ValueError("prior_strength_init must be non-negative")
        if tangent_scale <= 0 or tangent_norm_max <= 0:
            raise ValueError("tangent scale and norm limit must be positive")

        self.text_dim = int(text_dim)
        self.image_dim = int(image_dim)
        self.output_dim = int(output_dim)
        self.fusion_dim = int(fusion_dim)
        self.residual_scale = float(residual_scale)
        self.curvature_init = float(curvature_init)
        self.curvature_min = float(curvature_min)
        self.curvature_max = float(curvature_max)
        self.prior_strength_init = float(prior_strength_init)
        self.temperature_init = (
            math.sqrt(self.fusion_dim)
            if temperature_init is None
            else float(temperature_init)
        )
        self.tangent_scale = float(tangent_scale)
        self.tangent_norm_max = float(tangent_norm_max)
        self.initializer_seed = int(initializer_seed)
        if self.temperature_init <= 0:
            raise ValueError("temperature_init must be positive")

        curvature_fraction = (curvature_init - curvature_min) / (
            curvature_max - curvature_min
        )
        curvature_raw_init = _inverse_sigmoid(curvature_fraction)
        self.text_curvature_raw = self.add_weight(
            name="text_curvature_raw",
            shape=(),
            initializer=tf.keras.initializers.Constant(curvature_raw_init),
            trainable=True,
        )
        self.image_curvature_raw = self.add_weight(
            name="image_curvature_raw",
            shape=(),
            initializer=tf.keras.initializers.Constant(curvature_raw_init),
            trainable=True,
        )
        self.temperature_raw = self.add_weight(
            name="temperature_raw",
            shape=(),
            initializer=tf.keras.initializers.Constant(
                _inverse_softplus(self.temperature_init)
            ),
            trainable=True,
        )
        self.prior_strength_raw = self.add_weight(
            name="prior_strength_raw",
            shape=(),
            initializer=tf.keras.initializers.Constant(
                _inverse_softplus(max(self.prior_strength_init, 1e-6))
            ),
            trainable=True,
        )

        self.text_projection = tf.keras.layers.Dense(
            self.fusion_dim,
            kernel_initializer=tf.keras.initializers.GlorotUniform(
                seed=self.initializer_seed
            ),
            name="text_projection",
        )
        self.image_projection = tf.keras.layers.Dense(
            self.fusion_dim,
            kernel_initializer=tf.keras.initializers.GlorotUniform(
                seed=self.initializer_seed + 1
            ),
            name="image_projection",
        )
        self.query_projection = tf.keras.layers.Dense(
            self.fusion_dim,
            activation="tanh",
            kernel_initializer=tf.keras.initializers.GlorotUniform(
                seed=self.initializer_seed + 2
            ),
            name="query_projection",
        )
        self.output_projection = tf.keras.layers.Dense(
            self.output_dim,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="output_projection",
        )

        # Last-batch observability without changing the model's return signature.
        self._diag_text_weight = self.add_weight(
            name="diag_text_weight",
            shape=(),
            initializer="zeros",
            trainable=False,
        )
        self._diag_image_weight = self.add_weight(
            name="diag_image_weight",
            shape=(),
            initializer="zeros",
            trainable=False,
        )
        self._diag_attention_entropy = self.add_weight(
            name="diag_attention_entropy",
            shape=(),
            initializer="zeros",
            trainable=False,
        )
        self._diag_residual_ratio = self.add_weight(
            name="diag_residual_ratio",
            shape=(),
            initializer="zeros",
            trainable=False,
        )
        self._diag_manifold_error = self.add_weight(
            name="diag_manifold_error",
            shape=(),
            initializer="zeros",
            trainable=False,
        )

    def curvatures(self):
        span = self.curvature_max - self.curvature_min
        text = self.curvature_min + span * tf.sigmoid(self.text_curvature_raw)
        image = self.curvature_min + span * tf.sigmoid(self.image_curvature_raw)
        return text, image

    def _check_inputs(self, text_feature, image_feature, base_feature):
        for name, tensor, width in (
            ("text_feature", text_feature, self.text_dim),
            ("image_feature", image_feature, self.image_dim),
            ("base_feature", base_feature, self.output_dim),
        ):
            tf.debugging.assert_type(tensor, tf.float32, message=name)
            tf.debugging.assert_rank(tensor, 2, message=f"{name} must have shape [B, D]")
            tf.debugging.assert_equal(
                tf.shape(tensor)[1], width, message=f"{name} feature width mismatch"
            )
            tf.debugging.assert_all_finite(tensor, message=f"{name} contains NaN or Inf")
        batch_size = tf.shape(base_feature)[0]
        tf.debugging.assert_positive(batch_size, message="module B received an empty batch")
        tf.debugging.assert_equal(
            tf.shape(text_feature)[0], batch_size, message="text/base batch mismatch"
        )
        tf.debugging.assert_equal(
            tf.shape(image_feature)[0], batch_size, message="image/base batch mismatch"
        )

    def call(self, text_feature, image_feature, base_feature, update_diagnostics=True):
        text_feature = tf.convert_to_tensor(text_feature)
        image_feature = tf.convert_to_tensor(image_feature)
        base_feature = tf.convert_to_tensor(base_feature)
        self._check_inputs(text_feature, image_feature, base_feature)

        text_tangent = tf.nn.l2_normalize(
            self.text_projection(text_feature), axis=-1
        ) * self.tangent_scale
        image_tangent = tf.nn.l2_normalize(
            self.image_projection(image_feature), axis=-1
        ) * self.tangent_scale

        text_curvature, image_curvature = self.curvatures()
        shared_curvature = 0.5 * (text_curvature + image_curvature)
        text_point = expmap0(
            text_tangent,
            text_curvature,
            max_norm=self.tangent_norm_max,
        )
        image_point = expmap0(
            image_tangent,
            image_curvature,
            max_norm=self.tangent_norm_max,
        )
        text_shared = project_curvature(
            text_point,
            text_curvature,
            shared_curvature,
            max_norm=self.tangent_norm_max,
        )
        image_shared = project_curvature(
            image_point,
            image_curvature,
            shared_curvature,
            max_norm=self.tangent_norm_max,
        )

        text_shared_tangent = logmap0(text_shared, shared_curvature)
        image_shared_tangent = logmap0(image_shared, shared_curvature)
        query_tangent = self.query_projection(
            tf.concat([text_shared_tangent, image_shared_tangent], axis=-1)
        )
        query_point = expmap0(
            query_tangent,
            shared_curvature,
            max_norm=self.tangent_norm_max,
        )

        modality_points = tf.stack([text_shared, image_shared], axis=1)
        distances = pairwise_distance(query_point, modality_points, shared_curvature)
        temperature = tf.nn.softplus(self.temperature_raw) + 1e-6
        prior_strength = tf.nn.softplus(self.prior_strength_raw)
        curvature_prior = prior_strength * tf.math.log(
            tf.stack([text_curvature, image_curvature]) + 1e-6
        )
        scores = -tf.square(distances) / temperature + curvature_prior[tf.newaxis, :]
        weights = tf.nn.softmax(scores, axis=-1)
        tf.debugging.assert_all_finite(weights, message="module B weights contain NaN or Inf")
        tf.debugging.assert_near(
            tf.reduce_sum(weights, axis=-1),
            tf.ones(tf.shape(weights)[0], dtype=weights.dtype),
            atol=1e-5,
            message="module B weights do not sum to one",
        )

        fused_point = normalized_centroid(
            modality_points, weights, shared_curvature
        )
        fused_tangent = logmap0(fused_point, shared_curvature)
        delta = self.output_projection(fused_tangent)
        delta = tf.debugging.check_numerics(delta, "module B residual contains NaN or Inf")
        scaled_delta = tf.cast(self.residual_scale, delta.dtype) * delta
        output = base_feature + scaled_delta
        output = tf.debugging.check_numerics(output, "module B output contains NaN or Inf")
        tf.debugging.assert_equal(
            tf.shape(output), tf.shape(base_feature), message="module B changed output shape"
        )

        entropy = -tf.reduce_sum(
            weights * tf.math.log(tf.maximum(weights, 1e-8)), axis=-1
        )
        residual_ratio = tf.reduce_mean(
            tf.math.divide_no_nan(
                tf.norm(scaled_delta, axis=-1),
                tf.norm(base_feature, axis=-1) + 1e-6,
            )
        )
        manifold_error = tf.reduce_max(
            tf.stack(
                [
                    constraint_error(text_shared, shared_curvature),
                    constraint_error(image_shared, shared_curvature),
                    constraint_error(fused_point, shared_curvature),
                ],
                axis=0,
            )
        )
        if update_diagnostics:
            self._diag_text_weight.assign(tf.reduce_mean(weights[:, 0]))
            self._diag_image_weight.assign(tf.reduce_mean(weights[:, 1]))
            self._diag_attention_entropy.assign(tf.reduce_mean(entropy))
            self._diag_residual_ratio.assign(residual_ratio)
            self._diag_manifold_error.assign(manifold_error)
        return output

    def diagnostics(self):
        text_curvature, image_curvature = self.curvatures()
        return {
            "text_curvature": text_curvature,
            "image_curvature": image_curvature,
            "text_weight": self._diag_text_weight,
            "image_weight": self._diag_image_weight,
            "attention_entropy": self._diag_attention_entropy,
            "residual_ratio": self._diag_residual_ratio,
            "manifold_error": self._diag_manifold_error,
        }

    def get_config(self):
        config = super(CurvatureAwareFusion, self).get_config()
        config.update(
            {
                "text_dim": self.text_dim,
                "image_dim": self.image_dim,
                "output_dim": self.output_dim,
                "fusion_dim": self.fusion_dim,
                "residual_scale": self.residual_scale,
                "curvature_init": self.curvature_init,
                "curvature_min": self.curvature_min,
                "curvature_max": self.curvature_max,
                "prior_strength_init": self.prior_strength_init,
                "temperature_init": self.temperature_init,
                "tangent_scale": self.tangent_scale,
                "tangent_norm_max": self.tangent_norm_max,
                "initializer_seed": self.initializer_seed,
            }
        )
        return config
