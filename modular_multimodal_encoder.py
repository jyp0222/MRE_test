"""Opt-in multimodal encoder wrapper for paper-B-style curvature fusion."""

import tensorflow as tf

from curvature_fusion import CurvatureAwareFusion


class ModuleBMultimodalEncoder(tf.keras.models.Model):
    """Reuse the baseline encoders and add module B after feature extraction."""

    def __init__(self, base_encoder, **module_b_kwargs):
        super(ModuleBMultimodalEncoder, self).__init__(name="module_b_multimodal_encoder")
        self.base_encoder = base_encoder
        self.module_b = CurvatureAwareFusion(**module_b_kwargs)

    def _encode_view(self, h_cls, h_text_entity, image, update_diagnostics):
        image_tokens = self.base_encoder.image_encoder(image)
        image_feature = tf.reduce_mean(image_tokens, axis=1)
        text_feature = tf.concat([h_cls, h_text_entity], axis=-1)
        base_feature = tf.concat([text_feature, image_feature], axis=-1)

        tf.debugging.assert_type(base_feature, tf.float32, message="base feature")
        tf.debugging.assert_rank(base_feature, 2, message="base feature must be [B, 3072]")
        tf.debugging.assert_equal(
            tf.shape(text_feature)[1], 2304, message="module B expects text width 2304"
        )
        tf.debugging.assert_equal(
            tf.shape(image_feature)[1], 768, message="module B expects image width 768"
        )
        tf.debugging.assert_equal(
            tf.shape(base_feature)[1], 3072, message="DAEO expects encoder width 3072"
        )
        return self.module_b(
            text_feature,
            image_feature,
            base_feature,
            update_diagnostics=update_diagnostics,
        )

    @tf.function
    def call(self, s_ind, s_seg, head_idx, tail_idx, img, aug_flag=False):
        h_sentence, h_text_entity = self.base_encoder.sentence_encoder(
            s_ind, s_seg, head_idx, tail_idx
        )
        h_cls = h_sentence[:, 0, :]
        h_entity = self._encode_view(
            h_cls, h_text_entity, img, update_diagnostics=True
        )
        if aug_flag:
            h_aug_entity = self._encode_view(
                h_cls,
                h_text_entity,
                tf.image.flip_left_right(img),
                update_diagnostics=False,
            )
            return h_entity, h_aug_entity
        return h_entity

    def module_b_diagnostics(self):
        return self.module_b.diagnostics()
