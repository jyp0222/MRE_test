import tensorflow as tf
from vit_keras import vit
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import initializers
from tensorflow.keras.models import Sequential


class SimpleMultimodalEncoder(tf.keras.models.Model):
    def __init__(self, sentence_encoder):
        super(SimpleMultimodalEncoder, self).__init__()
        self.sentence_encoder = sentence_encoder
        self.image_encoder = vit.vit_b16(
            image_size=(384, 384),
            pretrained=True,
            include_top=False,
            pretrained_top=False,
        )
        for l in self.image_encoder.layers:
            l.trainable = False

    @tf.function
    def call(
        self,
        s_ind,
        s_seg,
        head_idx,
        tail_idx,
        img,
        aug_flag=False,
    ):
        h_sentence, h_text_entity = self.sentence_encoder(
            s_ind, s_seg, head_idx, tail_idx
        )
        h_cls = h_sentence[:, 0, :]
        h_image = self.image_encoder(img)
        h_image = tf.reduce_mean(h_image, axis=1)
        h_entity = tf.concat([h_cls, h_text_entity, h_image], axis=-1)
        if aug_flag:
            aug_img = tf.image.flip_left_right(img)
            h_aug_img = self.image_encoder(aug_img)
            h_aug_img = tf.reduce_mean(h_aug_img, axis=1)
            h_aug_entity = tf.concat([h_cls, h_text_entity, h_aug_img], axis=-1)
            return h_entity, h_aug_entity
        else:
            return h_entity