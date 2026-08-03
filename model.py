import tensorflow as tf
from framework import GCDMNetModel
from tensorflow.keras import initializers
import tensorflow_addons as tfa
from tensorflow.keras.layers import Dense, Dropout, Lambda
from tensorflow.keras.models import Sequential

class Autoencoder(tf.keras.models.Model):
    def __init__(self, input_size, hidden_size):
        super(Autoencoder, self).__init__()
        self.encoder = Dense(hidden_size)
        self.decoder = Dense(input_size)

    @tf.function
    def encode(self, x):
        h = self.encoder(x)
        return h

    @tf.function
    def call(self, x):
        h = self.encoder(x)
        r_x = self.decoder(h)
        loss = tf.losses.mean_squared_error(r_x, x)
        return h, loss

class SAE(GCDMNetModel):
    def __init__(self, encoder, K, input_size: int = 768 * 2, use_img: bool = True):
        GCDMNetModel.__init__(self, encoder, use_img)
        # self.K = K
        self.dropout = Dropout(0.1)
        self.ae1 = Autoencoder(input_size, 768)
        self.ae2 = Autoencoder(768, 512)
        self.ae3 = Autoencoder(512, 256)
        self.fc = Dense(K, activation="softmax")
        # self.cluster = self.add_weight(
        #     name="cluster",
        #     shape=(K, 256),
        #     initializer=initializers.GlorotNormal(),
        #     trainable=True,
        # )

    # @tf.function
    # def _batch_dist(self, a, b):
    #     return tf.matmul(a, b, transpose_b=True)

    @tf.function
    def predict(self, data):
        data = self.unpack_data(data)
        h_entity = self.encoder(*data)
        h_entity = self.ae1.encode(h_entity)
        h_entity = self.ae2.encode(h_entity)
        h_entity = self.ae3.encode(h_entity)
        prob = self.fc(h_entity)
        return tf.argmax(prob, axis=-1)

    @tf.function
    def _train_with_labeled_data(self, data, label):
        h_entity = self.encoder(*data)
        h_entity, ae_loss1 = self.ae1(h_entity)
        h_entity, ae_loss2 = self.ae2(h_entity)
        h_entity, ae_loss3 = self.ae3(h_entity)
        ae_loss = ae_loss1 + ae_loss2 + ae_loss3
        prob = self.fc(self.dropout(h_entity))
        pred = tf.argmax(prob, axis=-1)
        ce_loss = tf.losses.sparse_categorical_crossentropy(label, prob)
        loss = ce_loss + ae_loss
        return loss, pred

    @tf.function
    def _train_with_unlabeled_data(self, data):
        h_entity = self.encoder(*data)
        h_entity, ae_loss1 = self.ae1(h_entity)
        h_entity, ae_loss2 = self.ae2(h_entity)
        h_entity, ae_loss3 = self.ae3(h_entity)
        ae_loss = ae_loss1 + ae_loss2 + ae_loss3
        prob = self.fc(self.dropout(h_entity))
        pred = tf.stop_gradient(tf.argmax(prob, axis=-1))
        ce_loss = tf.losses.sparse_categorical_crossentropy(pred, prob)
        loss = ae_loss + ce_loss
        return loss

    @tf.function
    def call(self, labeled_data, label, unlabeled_data):
        labeled_data, unlabeled_data = self.unpack_data(labeled_data), self.unpack_data(unlabeled_data)
        labeled_loss, pred = self._train_with_labeled_data(labeled_data, label)
        unlabeled_loss = self._train_with_unlabeled_data(unlabeled_data)
        return pred, labeled_loss, unlabeled_loss


class VariationalAutoencoder(tf.keras.models.Model):
    def __init__(self, input_size: int = 768 * 2, hidden_size: int = 768):
        super(VariationalAutoencoder, self).__init__()
        self.dense4decoder = Sequential([
            Dense(hidden_size, activation=tf.nn.leaky_relu),
            Dense(hidden_size, activation=tf.nn.leaky_relu),
            Dense(input_size)
        ])
        self.dense4mu = Dense(hidden_size)
        self.dense4logvar = Dense(hidden_size)

    @tf.function
    def kl_loss(self, mu, logvar):
        _kl_loss = -0.5 * tf.reduce_sum(
            1.0 + logvar - tf.math.square(mu) - tf.math.exp(logvar), axis=-1
        )
        return tf.reduce_mean(_kl_loss)

    @tf.function
    def encode(self, x):
        mu = self.dense4mu(x)
        logvar = self.dense4logvar(x)
        kl_loss = self.kl_loss(mu, logvar)
        var = tf.math.exp(logvar)
        epsilon = tf.random.normal(tf.shape(mu))
        N = mu + tf.math.sqrt(var) * epsilon
        return N, kl_loss

    @tf.function
    def call(self, x):
        N, kl_loss = self.encode(x)
        r_x = self.dense4decoder(N)
        aug_x = r_x + x
        return aug_x, kl_loss


class DAEO(GCDMNetModel):
    def __init__(self, encoder, K, hidden_size, use_img: bool = False):
        GCDMNetModel.__init__(self, encoder, use_img)
        self.K = K
        self.fc = Dense(hidden_size)
        self.vae = VariationalAutoencoder(input_size=hidden_size)

        self.cluster = self.add_weight(
            name="cluster",
            shape=(K, hidden_size),
            initializer=initializers.GlorotNormal(),
            trainable=True,
        )

    @tf.function
    def _batch_dist(self, a, b):
        a = tf.nn.l2_normalize(a, -1)
        b = tf.nn.l2_normalize(b, -1)
        return tf.matmul(a, b, transpose_b=True)

    @tf.function
    def predict(self, data):
        data = self.unpack_data(data)
        h_entity = self.encoder(*data)
        h_entity = self.fc(h_entity)
        logit = self._batch_dist(h_entity, self.cluster)
        prob = tf.nn.softmax(logit, axis=-1)
        return tf.argmax(prob, axis=-1)

    @tf.function
    def train_vae_with_labeled_data(self, data, label):
        h_entity = self.encoder(*data)
        h_aug_entity, kl_loss = self.vae(h_entity)
        h_entity = self.fc(h_entity)
        h_aug_entity = self.fc(h_aug_entity)
        logit = self._batch_dist(h_entity, self.cluster)
        prob = tf.nn.softmax(logit, axis=-1)
        aug_logit = self._batch_dist(h_aug_entity, self.cluster)
        aug_prob = tf.nn.softmax(aug_logit, axis=-1)
        aug_ce_loss = tf.losses.sparse_categorical_crossentropy(label, aug_prob)
        consis_loss = tf.reduce_mean(- tf.reduce_sum(prob * tf.math.log(aug_prob), axis=-1))
        loss = kl_loss + consis_loss - aug_ce_loss
        return loss

    @tf.function
    def _train_with_labeled_data(self, data, label):
        h_entity = self.encoder(*data)
        h_aug_entity, _ = self.vae(h_entity)
        h_entity = self.fc(h_entity)
        h_aug_entity = self.fc(h_aug_entity)
        logit = self._batch_dist(h_entity, self.cluster)
        aug_logit = self._batch_dist(h_aug_entity, self.cluster)
        prob = tf.nn.softmax(logit, axis=-1)
        pred = tf.argmax(prob, axis=-1)
        ce_loss = tf.losses.sparse_categorical_crossentropy(label, prob)
        aug_prob = tf.nn.softmax(aug_logit, axis=-1)
        aug_ce_loss = tf.losses.sparse_categorical_crossentropy(label, aug_prob)
        loss = ce_loss + aug_ce_loss
        return loss, pred

    @tf.function
    def _train_with_unlabeled_data(self, data):
        h_entity, h_aug_entity = self.encoder(*data, aug_flag=True)
        h_entity = self.fc(h_entity)
        h_aug_entity = self.fc(h_aug_entity)
        dist = self._batch_dist(h_entity, self.cluster)
        prob = tf.nn.softmax(dist, axis=-1)
        pseudo_dist = self._batch_dist(h_aug_entity, self.cluster)
        pseudo_prob = tf.nn.softmax(pseudo_dist, axis=-1)
        ce_loss = tf.reduce_mean(- tf.reduce_sum(pseudo_prob * tf.math.log(prob), axis=-1))
        entropy_loss = tf.reduce_sum(- prob * tf.math.log(prob))
        loss = ce_loss + entropy_loss
        return loss

    @tf.function
    def call(self, labeled_data, label, unlabeled_data):
        labeled_data, unlabeled_data = self.unpack_data(labeled_data), self.unpack_data(unlabeled_data)
        labeled_loss, pred = self._train_with_labeled_data(labeled_data, label)
        unlabeled_loss = self._train_with_unlabeled_data(unlabeled_data)
        return pred, labeled_loss, unlabeled_loss


class SimGCD(GCDMNetModel):
    def __init__(self, encoder, K, hidden_size, lamb: float = 0.5, use_img: bool = False):
        GCDMNetModel.__init__(self, encoder, use_img)
        self.K = K
        self.lamb = lamb
        self.fc = Dense(hidden_size)
            
        self.cluster = self.add_weight(
            name="cluster",
            shape=(K, hidden_size),
            initializer=initializers.GlorotNormal(),
            trainable=True,
        )

    @tf.function
    def _batch_dist(self, a, b):
        a = tf.nn.l2_normalize(a, -1)
        b = tf.nn.l2_normalize(b, -1)
        return tf.matmul(a, b, transpose_b=True)

    @tf.function
    def predict(self, data):
        data = self.unpack_data(data)
        h_entity = self.encoder(*data)
        h_entity = self.fc(h_entity)
        dist = self._batch_dist(h_entity, self.cluster)
        return tf.argmax(dist, axis=-1)

    @tf.function
    def _train_with_labeled_data(self, data, label):
        h_entity = self.encoder(*data)
        h_entity = self.fc(h_entity)
        dist = self._batch_dist(h_entity, self.cluster)
        pred = tf.argmax(dist, axis=-1)
        prob = tf.nn.softmax(dist, axis=-1)
        ce_loss = tf.losses.sparse_categorical_crossentropy(label, prob)

        sample_dist = self._batch_dist(h_entity, h_entity)
        scl_loss = tfa.losses.npairs_loss(label, sample_dist)
        loss = self.lamb * (ce_loss + scl_loss)
        return loss, pred

    @tf.function
    def _train_with_unlabeled_data(self, data):
        h_entity, h_aug_entity = self.encoder(*data, aug_flag=True)
        h_entity = self.fc(h_entity)
        h_aug_entity = self.fc(h_aug_entity)
        dist = self._batch_dist(h_entity, self.cluster)
        prob = tf.nn.softmax(dist, axis=-1)
        pseudo_dist = self._batch_dist(h_aug_entity, self.cluster)
        pseudo_prob = tf.nn.softmax(pseudo_dist, axis=-1)
        ce_loss = tf.reduce_mean(- tf.reduce_sum(pseudo_prob * tf.math.log(prob), axis=-1))
        mean_prob = tf.reduce_mean(prob + pseudo_prob, axis=0)
        entropy_loss = tf.reduce_sum(- mean_prob * tf.math.log(mean_prob))

        N = tf.shape(h_entity)[0]
        mask = tf.eye(N, N)
        diag_mask = tf.cast(mask, tf.bool)
        nondiag_mask = tf.cast(1.0 - mask, tf.bool)
        sample_dist = self._batch_dist(h_entity, h_aug_entity)
        pos_sample_dist = tf.reshape(tf.boolean_mask(sample_dist, diag_mask), (N, 1))
        aug_neg_sample_dist = tf.reshape(tf.boolean_mask(sample_dist, nondiag_mask), (N, N - 1))
        sample_dist = self._batch_dist(h_entity, h_entity)
        raw_neg_sample_dist = tf.reshape(tf.boolean_mask(sample_dist, nondiag_mask), (N, N - 1))
        logits = tf.concat([pos_sample_dist, aug_neg_sample_dist, raw_neg_sample_dist], axis=-1)
        prob = tf.nn.softmax(logits, axis=-1)
        cl_label = tf.zeros(N)
        ucl_loss = tf.losses.sparse_categorical_crossentropy(cl_label, prob)
        loss = (1.0 - self.lamb) * (ce_loss + entropy_loss + ucl_loss)
        return loss

    @tf.function
    def call(self, labeled_data, label, unlabeled_data):
        labeled_data, unlabeled_data = self.unpack_data(labeled_data), self.unpack_data(unlabeled_data)
        labeled_loss, pred = self._train_with_labeled_data(labeled_data, label)
        unlabeled_loss = self._train_with_unlabeled_data(unlabeled_data)
        return pred, labeled_loss, unlabeled_loss


class GCD(GCDMNetModel):
    def __init__(self, encoder, K, hidden_size, lamb: float = 0.5, use_img: bool = False):
        GCDMNetModel.__init__(self, encoder, use_img)
        self.K = K
        self.lamb = lamb
        self.fc = Dense(hidden_size)
        self.cluster = self.add_weight(
            name="cluster",
            shape=(K, hidden_size),
            initializer=initializers.GlorotNormal(),
            trainable=True,
        )

    @tf.function
    def _batch_dist(self, a, b):
        return tf.matmul(a, b, transpose_b=True)

    @tf.function
    def predict(self, data):
        data = self.unpack_data(data)
        h_entity = self.encoder(*data)
        h_entity = self.fc(h_entity)
        dist = self._batch_dist(h_entity, self.cluster)
        return tf.argmax(dist, axis=-1)

    @tf.function
    def _train_with_labeled_data(self, data, label):
        h_entity = self.encoder(*data)
        h_entity = self.fc(h_entity)
        dist = self._batch_dist(h_entity, self.cluster)
        pred = tf.argmax(dist, axis=-1)
        prob = tf.nn.softmax(dist, axis=-1)
        cluster_loss = tf.losses.sparse_categorical_crossentropy(label, prob)

        sample_dist = self._batch_dist(h_entity, h_entity)
        scl_loss = tfa.losses.npairs_loss(label, sample_dist)
        loss = cluster_loss + self.lamb * scl_loss
        return loss, pred

    @tf.function
    def _train_with_unlabeled_data(self, data):
        h_entity, h_aug_entity = self.encoder(*data, aug_flag=True)
        h_entity = self.fc(h_entity)
        h_aug_entity = self.fc(h_aug_entity)
        dist = self._batch_dist(h_entity, self.cluster)
        pred = tf.argmax(dist, axis=-1)
        label = tf.one_hot(pred, depth=self.K)
        c = label @ self.cluster
        cluster_loss = tf.losses.mean_squared_error(c, h_entity)

        N = tf.shape(h_entity)[0]
        mask = tf.eye(N, N)
        diag_mask = tf.cast(mask, tf.bool)
        nondiag_mask = tf.cast(1.0 - mask, tf.bool)
        sample_dist = self._batch_dist(h_entity, h_aug_entity)
        pos_sample_dist = tf.reshape(tf.boolean_mask(sample_dist, diag_mask), (N, 1))
        aug_neg_sample_dist = tf.reshape(tf.boolean_mask(sample_dist, nondiag_mask), (N, N - 1))
        sample_dist = self._batch_dist(h_entity, h_entity)
        raw_neg_sample_dist = tf.reshape(tf.boolean_mask(sample_dist, nondiag_mask), (N, N - 1))
        logits = tf.concat([pos_sample_dist, aug_neg_sample_dist, raw_neg_sample_dist], axis=-1)
        prob = tf.nn.softmax(logits, axis=-1)
        cl_label = tf.zeros(N)
        ucl_loss = tf.losses.sparse_categorical_crossentropy(cl_label, prob)
        loss = cluster_loss + (1.0 - self.lamb) * ucl_loss
        return loss

    @tf.function
    def call(self, labeled_data, label, unlabeled_data):
        labeled_data, unlabeled_data = self.unpack_data(labeled_data), self.unpack_data(unlabeled_data)
        labeled_loss, pred = self._train_with_labeled_data(labeled_data, label)
        unlabeled_loss = self._train_with_unlabeled_data(unlabeled_data)
        return pred, labeled_loss, unlabeled_loss


class UNO(GCDMNetModel):
    def __init__(self, encoder, n_category, use_img: bool = True):
        GCDMNetModel.__init__(self, encoder, use_img)
        self.fc = Dense(n_category)
        self.dropout = Dropout(0.1)
        self.n_category = n_category

    @tf.function
    def predict(self, data):
        data = self.unpack_data(data)
        h_entity = self.encoder(*data)
        logits = self.fc(h_entity)
        prob = tf.nn.softmax(logits, axis=-1)
        pred = tf.argmax(prob, axis=-1)
        return pred

    @tf.function
    def _train_with_labeled_data(self, data, label):
        h_entity = self.encoder(*data)
        logits = self.fc(self.dropout(h_entity))
        prob = tf.nn.softmax(logits, axis=-1)
        pred = tf.argmax(prob, axis=-1)
        loss = tf.losses.sparse_categorical_crossentropy(label, prob)
        return loss, pred
    
    @tf.function
    def _train_with_unlabeled_data(self, data):
        h_entity, h_aug_entity = self.encoder(*data, aug_flag=True)
        logits = self.fc(h_entity)
        prob = tf.nn.softmax(tf.stop_gradient(logits), axis=-1)
        pseudo_label = tf.argmax(prob, axis=-1)

        aug_logits = self.fc(self.dropout(h_aug_entity)) 
        aug_prob = tf.nn.softmax(aug_logits, axis=-1)
        loss = tf.losses.sparse_categorical_crossentropy(pseudo_label, aug_prob)
        return loss

    @tf.function
    def call(self, labeled_data, label, unlabeled_data):
        labeled_data, unlabeled_data = self.unpack_data(labeled_data), self.unpack_data(unlabeled_data)
        labeled_loss, pred = self._train_with_labeled_data(labeled_data, label)
        unlabeled_loss = self._train_with_unlabeled_data(unlabeled_data)
        return pred, labeled_loss, unlabeled_loss


class RankStats(GCDMNetModel):
    def __init__(self, encoder, n_base_category, n_category, use_img: bool = True):
        GCDMNetModel.__init__(self, encoder, use_img)
        self.fc = Dense(n_category)
        self.dropout = Dropout(0.1)
        self.n_category = n_category
        self.n_base_category = n_base_category

    @tf.function
    def predict(self, data):
        data = self.unpack_data(data)
        h_entity = self.encoder(*data)
        logits = self.fc(h_entity)
        prob = tf.nn.softmax(logits, axis=-1)
        pred = tf.argmax(prob, axis=-1)
        return pred

    @tf.function
    def _train_with_labeled_data(self, data, label):
        h_entity = self.encoder(*data)
        logits = self.fc(self.dropout(h_entity))
        logits = logits[:, :self.n_base_category]
        prob = tf.nn.softmax(logits, axis=-1)
        pred = tf.argmax(prob, axis=-1)
        loss = tf.losses.sparse_categorical_crossentropy(label, prob)
        return loss, pred
    
    @tf.function
    def _train_with_unlabeled_data(self, data):
        h_entity = self.encoder(*data)
        logits = self.fc(self.dropout(h_entity))
        prob = tf.nn.softmax(tf.stop_gradient(logits), axis=-1)
        pseudo_label = tf.one_hot(tf.argmax(prob, axis=-1), depth=self.n_category)
        loss = tf.losses.binary_crossentropy(pseudo_label, logits, from_logits=True)
        return loss

    @tf.function
    def call(self, labeled_data, label, unlabeled_data):
        labeled_data, unlabeled_data = self.unpack_data(labeled_data), self.unpack_data(unlabeled_data)
        labeled_loss, pred = self._train_with_labeled_data(labeled_data, label)
        unlabeled_loss = self._train_with_unlabeled_data(unlabeled_data)
        return pred, labeled_loss, unlabeled_loss


class KMeans(GCDMNetModel):
    def __init__(self, encoder, K, hidden_size, use_img: bool = False):
        GCDMNetModel.__init__(self, encoder, use_img)
        self.K = K
        self.cluster = self.add_weight(
            name="cluster",
            shape=(K, hidden_size),
            initializer=initializers.GlorotNormal(),
            trainable=True,
        )

    @tf.function
    def _batch_dist(self, a, b):
        return tf.matmul(a, b, transpose_b=True)

    @tf.function
    def predict(self, data):
        data = self.unpack_data(data)
        h_entity = self.encoder(*data)
        dist = self._batch_dist(h_entity, self.cluster)
        return tf.argmax(dist, axis=-1)

    @tf.function
    def _train_with_labeled_data(self, data, label):
        h_entity = self.encoder(*data)
        dist = self._batch_dist(h_entity, self.cluster)
        pred = tf.argmax(dist, axis=-1)
        prob = tf.nn.softmax(dist, axis=-1)
        loss = tf.losses.sparse_categorical_crossentropy(label, prob)
        return loss, pred
    
    @tf.function
    def _train_with_unlabeled_data(self, data):
        h_entity = self.encoder(*data)
        dist = self._batch_dist(h_entity, self.cluster)
        pred = tf.argmax(dist, axis=-1)
        label = tf.one_hot(pred, depth=self.K)
        c = label @ self.cluster
        loss = tf.losses.mean_squared_error(c, h_entity)
        return loss

    @tf.function
    def call(self, labeled_data, label, unlabeled_data):
        labeled_data, unlabeled_data = self.unpack_data(labeled_data), self.unpack_data(unlabeled_data)
        labeled_loss, pred = self._train_with_labeled_data(labeled_data, label)
        unlabeled_loss = self._train_with_unlabeled_data(unlabeled_data)
        return pred, labeled_loss, unlabeled_loss