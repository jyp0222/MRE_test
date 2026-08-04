"""Synthetic tests for the optional TensorFlow module B.

These tests deliberately avoid FewRel, BERT weights, and ViT downloads so they
can run before the full server smoke test.
"""

import numpy as np
import tensorflow as tf

from curvature_fusion import CurvatureAwareFusion
from hyperbolic_geometry import constraint_error, expmap0, logmap0, project_curvature
from modular_multimodal_encoder import ModuleBMultimodalEncoder


class HyperbolicGeometryTest(tf.test.TestCase):
    def test_exp_log_round_trip_and_constraint(self):
        tf.random.set_seed(7)
        tangent = tf.random.normal([4, 8], stddev=0.1)
        for curvature in (0.2, 1.0, 3.0):
            point = expmap0(tangent, curvature)
            recovered = logmap0(point, curvature)
            self.assertAllLess(constraint_error(point, curvature), 1e-4)
            self.assertAllClose(tangent, recovered, atol=2e-4, rtol=2e-4)

    def test_cross_curvature_projection_stays_on_target_manifold(self):
        tangent = tf.random.normal([3, 6], stddev=0.15, seed=11)
        source_point = expmap0(tangent, 0.4)
        target_point = project_curvature(source_point, 0.4, 1.7)
        self.assertAllLess(constraint_error(target_point, 1.7), 1e-4)


class CurvatureAwareFusionTest(tf.test.TestCase):
    def _inputs(self):
        tf.random.set_seed(13)
        text = tf.random.normal([4, 12])
        image = tf.random.normal([4, 8])
        base = tf.concat([text, image], axis=-1)
        return text, image, base

    def _layer(self, residual_scale=0.05):
        return CurvatureAwareFusion(
            text_dim=12,
            image_dim=8,
            output_dim=20,
            fusion_dim=6,
            residual_scale=residual_scale,
        )

    def test_shape_finiteness_and_diagnostics(self):
        text, image, base = self._inputs()
        layer = self._layer()
        output = layer(text, image, base)
        self.assertAllEqual(tf.shape(output), [4, 20])
        self.assertAllEqual(tf.math.is_finite(output), tf.ones_like(output, dtype=tf.bool))

        diagnostics = layer.diagnostics()
        self.assertAllClose(
            diagnostics["text_weight"] + diagnostics["image_weight"], 1.0, atol=1e-5
        )
        self.assertGreater(float(diagnostics["text_curvature"].numpy()), 0.0)
        self.assertGreater(float(diagnostics["image_curvature"].numpy()), 0.0)
        self.assertLess(float(diagnostics["manifold_error"].numpy()), 1e-3)

    def test_zero_residual_scale_is_elementwise_path_neutral(self):
        text, image, base = self._inputs()
        layer = self._layer(residual_scale=0.0)
        layer(text, image, base)
        layer.output_projection.kernel.assign(
            tf.random.normal(tf.shape(layer.output_projection.kernel), seed=17)
        )
        layer.output_projection.bias.assign(
            tf.ones_like(layer.output_projection.bias) * 0.25
        )
        output = layer(text, image, base)
        self.assertTrue(np.array_equal(output.numpy(), base.numpy()))

    def test_forward_backward_has_finite_connected_gradients(self):
        text, image, base = self._inputs()
        layer = self._layer(residual_scale=0.05)
        layer(text, image, base)
        layer.output_projection.kernel.assign(
            tf.random.normal(tf.shape(layer.output_projection.kernel), stddev=0.01, seed=19)
        )
        with tf.GradientTape() as tape:
            output = layer(text, image, base)
            loss = tf.reduce_mean(tf.square(output))
        gradients = tape.gradient(loss, layer.trainable_variables)
        disconnected = [
            variable.name
            for gradient, variable in zip(gradients, layer.trainable_variables)
            if gradient is None
        ]
        self.assertEqual(disconnected, [])
        for gradient in gradients:
            value = gradient.values if isinstance(gradient, tf.IndexedSlices) else gradient
            self.assertAllEqual(
                tf.math.is_finite(value), tf.ones_like(value, dtype=tf.bool)
            )

    def test_seeded_module_initializers_do_not_advance_global_rng(self):
        tf.random.set_seed(29)
        reference_dense = tf.keras.layers.Dense(5)
        reference_dense(tf.ones([2, 7]))
        reference_kernel = reference_dense.kernel.numpy().copy()

        tf.random.set_seed(29)
        text = tf.ones([2, 12])
        image = tf.ones([2, 8])
        base = tf.concat([text, image], axis=-1)
        self._layer(residual_scale=0.0)(text, image, base)
        candidate_dense = tf.keras.layers.Dense(5)
        candidate_dense(tf.ones([2, 7]))
        self.assertAllEqual(candidate_dense.kernel, reference_kernel)


class _DummySentenceEncoder(tf.keras.layers.Layer):
    def call(self, s_ind, s_seg, head_idx, tail_idx):
        del s_seg, head_idx, tail_idx
        batch_size = tf.shape(s_ind)[0]
        seq_len = tf.shape(s_ind)[1]
        sentence = tf.zeros([batch_size, seq_len, 768], dtype=tf.float32)
        entities = tf.zeros([batch_size, 1536], dtype=tf.float32)
        return sentence, entities


class _DummyImageEncoder(tf.keras.layers.Layer):
    def call(self, image):
        batch_size = tf.shape(image)[0]
        value = tf.reduce_mean(image, axis=[1, 2, 3], keepdims=True)
        value = tf.reshape(value, [batch_size, 1, 1])
        return tf.tile(value, [1, 4, 768])


class _DummyBaseEncoder(tf.keras.models.Model):
    def __init__(self):
        super(_DummyBaseEncoder, self).__init__()
        self.sentence_encoder = _DummySentenceEncoder()
        self.image_encoder = _DummyImageEncoder()


class ModuleBEncoderContractTest(tf.test.TestCase):
    def test_regular_and_augmented_contract(self):
        encoder = ModuleBMultimodalEncoder(
            _DummyBaseEncoder(), fusion_dim=8, residual_scale=0.0
        )
        batch_size = 2
        seq_len = 5
        s_ind = tf.zeros([batch_size, seq_len], dtype=tf.int64)
        s_seg = tf.zeros([batch_size, seq_len], dtype=tf.int64)
        head_idx = tf.zeros([batch_size, 1], dtype=tf.int64)
        tail_idx = tf.ones([batch_size, 1], dtype=tf.int64)
        image = tf.random.uniform([batch_size, 384, 384, 3], seed=23)

        output = encoder(s_ind, s_seg, head_idx, tail_idx, image)
        output_pair = encoder(
            s_ind, s_seg, head_idx, tail_idx, image, aug_flag=True
        )
        self.assertAllEqual(tf.shape(output), [batch_size, 3072])
        self.assertEqual(len(output_pair), 2)
        self.assertAllEqual(tf.shape(output_pair[0]), [batch_size, 3072])
        self.assertAllEqual(tf.shape(output_pair[1]), [batch_size, 3072])


if __name__ == "__main__":
    tf.test.main()
