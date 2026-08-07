"""TensorFlow adaptation of TEDFusion's attribute-text gating module.

The original ATGFM consumes paired infrared/visible attributes and generated
captions.  This task has one RGB image and relation text, so the transferable
part is the learned gate ``g * attribute + (1 - g) * text``.  The fused
condition modulates only DAEO's pooled image feature through a conservative
residual path.  No image-reconstruction or hyperbolic loss is introduced.
"""

import math

import tensorflow as tf


class AttributeTextGatedFusion(tf.keras.layers.Layer):
    """Fuse scene attributes and text while preserving DAEO's feature width."""

    def __init__(
        self,
        text_dim=2304,
        image_dim=768,
        output_dim=3072,
        gate_dim=64,
        residual_scale=0.01,
        attribute_image_size=96,
        initializer_seed=20000,
        name="module_c_attribute_text_gate",
        **kwargs,
    ):
        super(AttributeTextGatedFusion, self).__init__(name=name, **kwargs)
        if min(text_dim, image_dim, output_dim, gate_dim, attribute_image_size) <= 0:
            raise ValueError("all feature dimensions and image size must be positive")
        if attribute_image_size < 4:
            raise ValueError("attribute_image_size must be at least 4")
        if output_dim != text_dim + image_dim:
            raise ValueError("output_dim must equal text_dim + image_dim")
        if residual_scale < 0:
            raise ValueError("residual_scale must be non-negative")

        self.text_dim = int(text_dim)
        self.image_dim = int(image_dim)
        self.output_dim = int(output_dim)
        self.gate_dim = int(gate_dim)
        self.residual_scale = float(residual_scale)
        self.attribute_image_size = int(attribute_image_size)
        self.initializer_seed = int(initializer_seed)

        self.attribute_projection = tf.keras.layers.Dense(
            self.gate_dim,
            kernel_initializer=tf.keras.initializers.GlorotUniform(
                seed=self.initializer_seed
            ),
            name="attribute_projection",
        )
        self.text_projection = tf.keras.layers.Dense(
            self.gate_dim,
            kernel_initializer=tf.keras.initializers.GlorotUniform(
                seed=self.initializer_seed + 1
            ),
            name="text_projection",
        )
        self.fusion_gate = tf.keras.layers.Dense(
            self.gate_dim,
            activation="sigmoid",
            kernel_initializer=tf.keras.initializers.GlorotUniform(
                seed=self.initializer_seed + 2
            ),
            name="fusion_gate",
        )
        # Zero initialization makes the first enabled forward pass neutral.
        self.affine_projection = tf.keras.layers.Dense(
            self.image_dim * 2,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="image_affine_projection",
        )

        self._diag_attribute_weight = self.add_weight(
            name="diag_attribute_weight",
            shape=(),
            initializer="zeros",
            trainable=False,
        )
        self._diag_text_weight = self.add_weight(
            name="diag_text_weight",
            shape=(),
            initializer="zeros",
            trainable=False,
        )
        self._diag_gate_std = self.add_weight(
            name="diag_gate_std",
            shape=(),
            initializer="zeros",
            trainable=False,
        )
        self._diag_saturation = self.add_weight(
            name="diag_saturation",
            shape=(),
            initializer="zeros",
            trainable=False,
        )
        self._diag_brightness = self.add_weight(
            name="diag_brightness",
            shape=(),
            initializer="zeros",
            trainable=False,
        )
        self._diag_texture_proxy = self.add_weight(
            name="diag_texture_proxy",
            shape=(),
            initializer="zeros",
            trainable=False,
        )
        self._diag_contrast = self.add_weight(
            name="diag_contrast",
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

    def _check_image(self, image):
        tf.debugging.assert_type(image, tf.float32, message="image")
        tf.debugging.assert_rank(
            image, 4, message="image must have shape [B, H, W, 3]"
        )
        tf.debugging.assert_positive(
            tf.shape(image)[0], message="module C received an empty batch"
        )
        tf.debugging.assert_positive(
            tf.shape(image)[1], message="module C received an empty image height"
        )
        tf.debugging.assert_positive(
            tf.shape(image)[2], message="module C received an empty image width"
        )
        tf.debugging.assert_equal(
            tf.shape(image)[3], 3, message="module C expects RGB images"
        )
        tf.debugging.assert_all_finite(image, message="image contains NaN or Inf")
        tolerance = tf.cast(1e-3, image.dtype)
        tf.debugging.assert_greater_equal(
            tf.reduce_min(image),
            tf.cast(-1.0, image.dtype) - tolerance,
            message="module C expects ViT-preprocessed images in [-1, 1]",
        )
        tf.debugging.assert_less_equal(
            tf.reduce_max(image),
            tf.cast(1.0, image.dtype) + tolerance,
            message="module C expects ViT-preprocessed images in [-1, 1]",
        )

    def _check_inputs(self, text_feature, image_feature, image, base_feature):
        for name, tensor, width in (
            ("text_feature", text_feature, self.text_dim),
            ("image_feature", image_feature, self.image_dim),
            ("base_feature", base_feature, self.output_dim),
        ):
            tf.debugging.assert_type(tensor, tf.float32, message=name)
            tf.debugging.assert_rank(tensor, 2, message=name + " must have shape [B, D]")
            tf.debugging.assert_equal(
                tf.shape(tensor)[1], width, message=name + " feature width mismatch"
            )
            tf.debugging.assert_all_finite(
                tensor, message=name + " contains NaN or Inf"
            )
        self._check_image(image)
        batch_size = tf.shape(base_feature)[0]
        for name, tensor in (
            ("text_feature", text_feature),
            ("image_feature", image_feature),
            ("image", image),
        ):
            tf.debugging.assert_equal(
                tf.shape(tensor)[0], batch_size, message=name + "/base batch mismatch"
            )

    def extract_attributes(self, image):
        """Return per-image ``[saturation, brightness, texture, contrast]``.

        TEDFusion computes texture with a CPU GLCM pipeline.  To keep this
        module inside ``tf.function`` and avoid changing the dataloader, the
        texture entry is a documented multi-scale structural proxy: normalized
        Sobel energy after a small attribute-only resize.  The other three
        attributes follow the paper definitions directly.
        """

        image = tf.convert_to_tensor(image)
        self._check_image(image)
        return self._extract_attributes(image)

    def _extract_attributes(self, image):
        """Extract attributes after the caller has validated the image."""

        image_01 = tf.clip_by_value((image + 1.0) * 0.5, 0.0, 1.0)
        hsv = tf.image.rgb_to_hsv(image_01)
        saturation = tf.reduce_mean(hsv[:, :, :, 1], axis=[1, 2])
        brightness = tf.reduce_mean(hsv[:, :, :, 2], axis=[1, 2])

        channel_mean = tf.reduce_mean(image_01, axis=[1, 2], keepdims=True)
        channel_variance = tf.reduce_mean(
            tf.square(image_01 - channel_mean), axis=[1, 2]
        )
        channel_std = tf.sqrt(
            tf.maximum(channel_variance, tf.zeros_like(channel_variance))
        )
        contrast = tf.reduce_mean(channel_std, axis=-1)

        gray = tf.image.rgb_to_grayscale(image_01)
        texture_levels = []
        for size in (
            self.attribute_image_size,
            max(self.attribute_image_size // 2, 2),
            max(self.attribute_image_size // 4, 2),
        ):
            texture_gray = tf.image.resize(gray, [size, size])
            sobel = tf.image.sobel_edges(texture_gray)
            gradient_magnitude = tf.sqrt(
                tf.maximum(
                    tf.reduce_sum(tf.square(sobel), axis=-1),
                    tf.zeros_like(sobel[:, :, :, :, 0]),
                )
            )
            sobel_max = tf.cast(4.0 * math.sqrt(2.0), gradient_magnitude.dtype)
            texture_levels.append(
                tf.reduce_mean(gradient_magnitude, axis=[1, 2, 3]) / sobel_max
            )
        texture = tf.reduce_mean(tf.stack(texture_levels, axis=-1), axis=-1)

        attributes = tf.stack(
            [saturation, brightness, texture, contrast], axis=-1
        )
        attributes = tf.clip_by_value(attributes, 0.0, 1.0)
        tf.debugging.assert_all_finite(
            attributes, message="module C attributes contain NaN or Inf"
        )
        tf.debugging.assert_equal(
            tf.shape(attributes)[1], 4, message="module C must produce four attributes"
        )
        return tf.stop_gradient(attributes)

    def fuse_embeddings(self, attribute_embedding, text_embedding):
        """Apply TEDFusion's ``g*a + (1-g)*t`` gating equation."""

        gate = self.fusion_gate(
            tf.concat([attribute_embedding, text_embedding], axis=-1)
        )
        fused = gate * attribute_embedding + (1.0 - gate) * text_embedding
        return fused, gate

    def call(
        self,
        text_feature,
        image_feature,
        image,
        base_feature,
        update_diagnostics=True,
    ):
        text_feature = tf.convert_to_tensor(text_feature)
        image_feature = tf.convert_to_tensor(image_feature)
        image = tf.convert_to_tensor(image)
        base_feature = tf.convert_to_tensor(base_feature)
        self._check_inputs(text_feature, image_feature, image, base_feature)

        attributes = self._extract_attributes(image)
        attribute_embedding = self.attribute_projection(attributes)
        text_embedding = self.text_projection(text_feature)
        fused_condition, gate = self.fuse_embeddings(
            attribute_embedding, text_embedding
        )
        affine = self.affine_projection(fused_condition)
        gamma, beta = tf.split(affine, 2, axis=-1)
        image_delta = gamma * image_feature + beta
        residual = tf.concat([tf.zeros_like(text_feature), image_delta], axis=-1)
        scaled_residual = tf.cast(self.residual_scale, residual.dtype) * residual
        output = base_feature + scaled_residual

        tf.debugging.assert_all_finite(gate, message="module C gate contains NaN or Inf")
        tf.debugging.assert_all_finite(
            output, message="module C output contains NaN or Inf"
        )
        tf.debugging.assert_equal(
            tf.shape(output), tf.shape(base_feature), message="module C changed output shape"
        )

        attribute_weight = tf.reduce_mean(gate)
        text_weight = tf.reduce_mean(1.0 - gate)
        gate_variance = tf.reduce_mean(tf.square(gate - attribute_weight))
        gate_std = tf.sqrt(tf.maximum(gate_variance, 0.0))
        residual_ratio = tf.reduce_mean(
            tf.math.divide_no_nan(
                tf.norm(tf.cast(self.residual_scale, image_delta.dtype) * image_delta, axis=-1),
                tf.norm(image_feature, axis=-1) + 1e-6,
            )
        )
        if update_diagnostics:
            self._diag_attribute_weight.assign(attribute_weight)
            self._diag_text_weight.assign(text_weight)
            self._diag_gate_std.assign(gate_std)
            self._diag_saturation.assign(tf.reduce_mean(attributes[:, 0]))
            self._diag_brightness.assign(tf.reduce_mean(attributes[:, 1]))
            self._diag_texture_proxy.assign(tf.reduce_mean(attributes[:, 2]))
            self._diag_contrast.assign(tf.reduce_mean(attributes[:, 3]))
            self._diag_residual_ratio.assign(residual_ratio)
        return output

    def diagnostics(self):
        return {
            "attribute_weight": self._diag_attribute_weight,
            "text_weight": self._diag_text_weight,
            "gate_std": self._diag_gate_std,
            "saturation": self._diag_saturation,
            "brightness": self._diag_brightness,
            "texture_proxy": self._diag_texture_proxy,
            "contrast": self._diag_contrast,
            "residual_ratio": self._diag_residual_ratio,
        }

    def get_config(self):
        config = super(AttributeTextGatedFusion, self).get_config()
        config.update(
            {
                "text_dim": self.text_dim,
                "image_dim": self.image_dim,
                "output_dim": self.output_dim,
                "gate_dim": self.gate_dim,
                "residual_scale": self.residual_scale,
                "attribute_image_size": self.attribute_image_size,
                "initializer_seed": self.initializer_seed,
            }
        )
        return config
