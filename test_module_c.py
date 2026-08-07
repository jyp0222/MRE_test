"""Synthetic tests for the optional TensorFlow module C.

The suite does not require FewRel, BERT weights, or a ViT checkpoint.
"""

import math

import numpy as np
import tensorflow as tf

from attribute_text_gate import AttributeTextGatedFusion
from module_c_multimodal_encoder import ModuleCMultimodalEncoder
from vit_keras import vit


class AttributeTextGatedFusionTest(tf.test.TestCase):
    def _layer(self, residual_scale=0.01):
        return AttributeTextGatedFusion(
            text_dim=12,
            image_dim=8,
            output_dim=20,
            gate_dim=4,
            residual_scale=residual_scale,
            attribute_image_size=16,
            initializer_seed=41,
        )

    def _inputs(self):
        tf.random.set_seed(13)
        text = tf.random.normal([4, 12])
        image_feature = tf.random.normal([4, 8])
        image = tf.random.uniform([4, 16, 16, 3], minval=-1.0, maxval=1.0)
        base = tf.concat([text, image_feature], axis=-1)
        return text, image_feature, image, base

    def test_vit_range_and_known_color_attributes(self):
        raw_range = tf.constant([0.0, 127.5, 255.0], dtype=tf.float32)
        self.assertAllClose(vit.preprocess_inputs(raw_range), [-1.0, 0.0, 1.0])

        colors = tf.constant(
            [
                [1.0, 0.0, 0.0],
                [0.5, 0.5, 0.5],
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=tf.float32,
        )
        image_01 = tf.tile(colors[:, None, None, :], [1, 16, 16, 1])
        attributes = self._layer().extract_attributes(image_01 * 2.0 - 1.0)
        expected = tf.constant(
            [
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=tf.float32,
        )
        self.assertAllClose(attributes, expected, atol=1e-6, rtol=1e-6)

        step = tf.concat(
            [tf.zeros([16, 8]), tf.ones([16, 8])], axis=1
        )
        step = tf.tile(step[None, :, :, None], [1, 1, 1, 3])
        step_attributes = self._layer().extract_attributes(
            step * 2.0 - 1.0
        )[0]
        self.assertAllClose(step_attributes[0], 0.0, atol=1e-6)
        self.assertAllClose(step_attributes[1], 0.5, atol=1e-6)
        self.assertGreater(float(step_attributes[2].numpy()), 0.0)
        self.assertAllClose(step_attributes[3], 0.5, atol=1e-6)

    def test_attributes_are_per_example_and_flip_invariant(self):
        horizontal = tf.linspace(-1.0, 1.0, 16)
        image_a = tf.tile(horizontal[None, None, :, None], [1, 16, 1, 3])
        image_b = -tf.ones([1, 16, 16, 3], dtype=tf.float32)
        layer = self._layer()
        separate = layer.extract_attributes(image_a)
        batched = layer.extract_attributes(tf.concat([image_a, image_b], axis=0))
        flipped = layer.extract_attributes(tf.image.flip_left_right(image_a))
        self.assertAllClose(separate[0], batched[0], atol=1e-6, rtol=1e-6)
        self.assertAllClose(separate, flipped, atol=1e-5, rtol=1e-5)

    def test_gate_formula_uses_attribute_weight(self):
        layer = self._layer()
        attribute_embedding = tf.constant([[1.0, 2.0, 3.0, 4.0]])
        text_embedding = tf.constant([[5.0, 6.0, 7.0, 8.0]])
        layer.fuse_embeddings(attribute_embedding, text_embedding)
        layer.fusion_gate.kernel.assign(tf.zeros_like(layer.fusion_gate.kernel))
        gate_value = 0.25
        gate_bias = math.log(gate_value / (1.0 - gate_value))
        layer.fusion_gate.bias.assign(
            tf.ones_like(layer.fusion_gate.bias) * gate_bias
        )
        fused, gate = layer.fuse_embeddings(attribute_embedding, text_embedding)
        expected = gate_value * attribute_embedding + (1.0 - gate_value) * text_embedding
        self.assertAllClose(gate, tf.ones_like(gate) * gate_value, atol=1e-6)
        self.assertAllClose(fused, expected, atol=1e-6)

    def test_shape_finiteness_and_diagnostics(self):
        text, image_feature, image, base = self._inputs()
        layer = self._layer()
        output = layer(text, image_feature, image, base)
        self.assertAllEqual(tf.shape(output), [4, 20])
        self.assertEqual(output.dtype, tf.float32)
        self.assertAllEqual(tf.math.is_finite(output), tf.ones_like(output, dtype=tf.bool))

        diagnostics = layer.diagnostics()
        self.assertAllClose(
            diagnostics["attribute_weight"] + diagnostics["text_weight"],
            1.0,
            atol=1e-5,
        )
        for name in ("saturation", "brightness", "texture_proxy", "contrast"):
            self.assertGreaterEqual(float(diagnostics[name].numpy()), 0.0)
            self.assertLessEqual(float(diagnostics[name].numpy()), 1.0)

    def test_zero_residual_scale_is_elementwise_path_neutral(self):
        text, image_feature, image, base = self._inputs()
        layer = self._layer(residual_scale=0.0)
        layer(text, image_feature, image, base)
        layer.affine_projection.kernel.assign(
            tf.random.normal(tf.shape(layer.affine_projection.kernel), seed=17)
        )
        layer.affine_projection.bias.assign(
            tf.ones_like(layer.affine_projection.bias) * 0.25
        )
        output = layer(text, image_feature, image, base)
        self.assertTrue(np.array_equal(output.numpy(), base.numpy()))

    def test_forward_backward_has_finite_connected_gradients(self):
        text, image_feature, image, base = self._inputs()
        layer = self._layer(residual_scale=0.01)
        layer(text, image_feature, image, base)
        layer.affine_projection.kernel.assign(
            tf.random.normal(
                tf.shape(layer.affine_projection.kernel), stddev=0.01, seed=19
            )
        )
        with tf.GradientTape() as tape:
            output = layer(text, image_feature, image, base)
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
        for layer_name in (
            "attribute_projection",
            "text_projection",
            "fusion_gate",
            "image_affine_projection",
        ):
            group = [
                gradient
                for gradient, variable in zip(gradients, layer.trainable_variables)
                if layer_name in variable.name
            ]
            self.assertTrue(group)
            self.assertGreater(float(tf.linalg.global_norm(group).numpy()), 0.0)

    def test_zero_initialized_first_step_keeps_upstream_gradients_connected(self):
        text, image_feature, image, base = self._inputs()
        layer = self._layer(residual_scale=0.01)
        with tf.GradientTape() as tape:
            output = layer(text, image_feature, image, base)
            loss = tf.reduce_mean(tf.square(output))
        gradients = tape.gradient(loss, layer.trainable_variables)
        self.assertEqual([gradient for gradient in gradients if gradient is None], [])
        for gradient in gradients:
            value = gradient.values if isinstance(gradient, tf.IndexedSlices) else gradient
            self.assertAllEqual(
                tf.math.is_finite(value), tf.ones_like(value, dtype=tf.bool)
            )

        upstream = [
            gradient
            for gradient, variable in zip(gradients, layer.trainable_variables)
            if "image_affine_projection" not in variable.name
        ]
        affine = [
            gradient
            for gradient, variable in zip(gradients, layer.trainable_variables)
            if "image_affine_projection" in variable.name
        ]
        self.assertAllClose(tf.linalg.global_norm(upstream), 0.0, atol=1e-8)
        self.assertGreater(float(tf.linalg.global_norm(affine).numpy()), 0.0)

    def test_seeded_module_initializers_do_not_advance_global_rng(self):
        tf.random.set_seed(29)
        reference_dense = tf.keras.layers.Dense(5)
        reference_dense(tf.ones([2, 7]))
        reference_kernel = reference_dense.kernel.numpy().copy()

        tf.random.set_seed(29)
        text = tf.ones([2, 12])
        image_feature = tf.ones([2, 8])
        image = tf.zeros([2, 16, 16, 3])
        base = tf.concat([text, image_feature], axis=-1)
        self._layer(residual_scale=0.0)(text, image_feature, image, base)
        candidate_dense = tf.keras.layers.Dense(5)
        candidate_dense(tf.ones([2, 7]))
        self.assertAllEqual(candidate_dense.kernel, reference_kernel)

    def test_zero_scale_call_does_not_advance_global_rng(self):
        text, image_feature, image, base = self._inputs()
        layer = self._layer(residual_scale=0.0)
        layer(text, image_feature, image, base)

        tf.random.set_seed(31)
        reference = tf.random.uniform([8])
        tf.random.set_seed(31)
        layer(text, image_feature, image, base)
        candidate = tf.random.uniform([8])
        self.assertAllEqual(candidate, reference)

    def test_non_finite_inputs_fail_fast(self):
        text, image_feature, image, base = self._inputs()
        layer = self._layer()
        bad_text = tf.tensor_scatter_nd_update(
            text, [[0, 0]], [tf.constant(float("inf"), dtype=tf.float32)]
        )
        bad_image_feature = tf.tensor_scatter_nd_update(
            image_feature, [[0, 0]], [tf.constant(float("nan"), dtype=tf.float32)]
        )
        bad_image = tf.tensor_scatter_nd_update(
            image, [[0, 0, 0, 0]], [tf.constant(float("nan"), dtype=tf.float32)]
        )
        bad_base = tf.tensor_scatter_nd_update(
            base, [[0, 0]], [tf.constant(float("inf"), dtype=tf.float32)]
        )
        for values in (
            (bad_text, image_feature, image, base),
            (text, bad_image_feature, image, base),
            (text, image_feature, bad_image, base),
            (text, image_feature, image, bad_base),
        ):
            with self.assertRaises(tf.errors.InvalidArgumentError):
                layer(*values)

    def test_invalid_configuration_and_image_range_fail_fast(self):
        with self.assertRaises(ValueError):
            AttributeTextGatedFusion(residual_scale=-0.1)
        with self.assertRaises(ValueError):
            AttributeTextGatedFusion(attribute_image_size=3)
        with self.assertRaises(tf.errors.InvalidArgumentError):
            self._layer().extract_attributes(tf.ones([1, 16, 16, 3]) * 1.1)


class _DummySentenceEncoder(tf.keras.layers.Layer):
    def call(self, s_ind, s_seg, head_idx, tail_idx):
        del s_seg, head_idx, tail_idx
        batch_size = tf.shape(s_ind)[0]
        seq_len = tf.shape(s_ind)[1]
        sentence = tf.zeros([batch_size, seq_len, 768], dtype=tf.float32)
        entities = tf.zeros([batch_size, 1536], dtype=tf.float32)
        return sentence, entities


class _FlipSensitiveImageEncoder(tf.keras.layers.Layer):
    def call(self, image):
        batch_size = tf.shape(image)[0]
        width_weights = tf.linspace(0.0, 1.0, tf.shape(image)[2])
        weighted = image * width_weights[None, None, :, None]
        value = tf.reduce_mean(weighted, axis=[1, 2, 3], keepdims=True)
        value = tf.reshape(value, [batch_size, 1, 1])
        return tf.tile(value, [1, 4, 768])


class _DummyBaseEncoder(tf.keras.models.Model):
    def __init__(self):
        super(_DummyBaseEncoder, self).__init__()
        self.sentence_encoder = _DummySentenceEncoder()
        self.image_encoder = _FlipSensitiveImageEncoder()


class ModuleCEncoderContractTest(tf.test.TestCase):
    def test_regular_and_augmented_contract_is_path_neutral_at_zero_scale(self):
        base_encoder = _DummyBaseEncoder()
        encoder = ModuleCMultimodalEncoder(
            base_encoder,
            gate_dim=8,
            residual_scale=0.0,
            attribute_image_size=16,
        )
        batch_size = 2
        seq_len = 5
        s_ind = tf.zeros([batch_size, seq_len], dtype=tf.int64)
        s_seg = tf.zeros([batch_size, seq_len], dtype=tf.int64)
        head_idx = tf.zeros([batch_size, 1], dtype=tf.int64)
        tail_idx = tf.ones([batch_size, 1], dtype=tf.int64)
        horizontal = tf.linspace(-1.0, 1.0, 16)
        image = tf.tile(
            horizontal[None, None, :, None], [batch_size, 16, 1, 3]
        )

        encoder(s_ind, s_seg, head_idx, tail_idx, image)
        encoder.module_c.affine_projection.kernel.assign(
            tf.random.normal(
                tf.shape(encoder.module_c.affine_projection.kernel), seed=53
            )
        )
        encoder.module_c.affine_projection.bias.assign(
            tf.ones_like(encoder.module_c.affine_projection.bias) * 0.25
        )
        output = encoder(s_ind, s_seg, head_idx, tail_idx, image)
        output_pair = encoder(
            s_ind, s_seg, head_idx, tail_idx, image, aug_flag=True
        )
        image_feature = tf.reduce_mean(base_encoder.image_encoder(image), axis=1)
        flipped_feature = tf.reduce_mean(
            base_encoder.image_encoder(tf.image.flip_left_right(image)), axis=1
        )
        text_feature = tf.zeros([batch_size, 2304], dtype=tf.float32)
        expected = tf.concat([text_feature, image_feature], axis=-1)
        expected_flipped = tf.concat([text_feature, flipped_feature], axis=-1)

        self.assertAllEqual(tf.shape(output), [batch_size, 3072])
        self.assertEqual(len(output_pair), 2)
        self.assertTrue(np.array_equal(output.numpy(), expected.numpy()))
        self.assertTrue(np.array_equal(output_pair[0].numpy(), expected.numpy()))
        self.assertTrue(
            np.array_equal(output_pair[1].numpy(), expected_flipped.numpy())
        )
        self.assertFalse(np.array_equal(expected.numpy(), expected_flipped.numpy()))

    def test_positive_scale_applies_module_to_regular_and_augmented_views(self):
        base_encoder = _DummyBaseEncoder()
        encoder = ModuleCMultimodalEncoder(
            base_encoder,
            gate_dim=8,
            residual_scale=0.1,
            attribute_image_size=16,
        )
        batch_size = 2
        seq_len = 5
        s_ind = tf.zeros([batch_size, seq_len], dtype=tf.int64)
        s_seg = tf.zeros([batch_size, seq_len], dtype=tf.int64)
        head_idx = tf.zeros([batch_size, 1], dtype=tf.int64)
        tail_idx = tf.ones([batch_size, 1], dtype=tf.int64)
        horizontal = tf.linspace(-1.0, 1.0, 16)
        image = tf.tile(
            horizontal[None, None, :, None], [batch_size, 16, 1, 3]
        )

        encoder(s_ind, s_seg, head_idx, tail_idx, image)
        encoder.module_c.affine_projection.kernel.assign(
            tf.zeros_like(encoder.module_c.affine_projection.kernel)
        )
        encoder.module_c.affine_projection.bias.assign(
            tf.ones_like(encoder.module_c.affine_projection.bias)
        )
        output = encoder(s_ind, s_seg, head_idx, tail_idx, image)
        output_pair = encoder(
            s_ind, s_seg, head_idx, tail_idx, image, aug_flag=True
        )

        text_feature = tf.zeros([batch_size, 2304], dtype=tf.float32)
        baseline = tf.concat(
            [
                text_feature,
                tf.reduce_mean(base_encoder.image_encoder(image), axis=1),
            ],
            axis=-1,
        )
        flipped_baseline = tf.concat(
            [
                text_feature,
                tf.reduce_mean(
                    base_encoder.image_encoder(tf.image.flip_left_right(image)),
                    axis=1,
                ),
            ],
            axis=-1,
        )
        self.assertFalse(np.array_equal(output.numpy(), baseline.numpy()))
        self.assertFalse(
            np.array_equal(output_pair[1].numpy(), flipped_baseline.numpy())
        )
        self.assertAllEqual(output, output_pair[0])
        self.assertGreater(
            float(encoder.module_c_diagnostics()["residual_ratio"].numpy()), 0.0
        )


if __name__ == "__main__":
    tf.test.main()
