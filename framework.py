import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from scipy.optimize import linear_sum_assignment as linear_assignment
from rich.progress import (
    SpinnerColumn,
    Progress,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

class GCDMNetModel(tf.keras.models.Model):
    def __init__(self, encoder, use_img):
        super(GCDMNetModel, self).__init__()
        self.encoder = encoder
        self.use_img = use_img

    def unpack_data(self, data):
        return data[:-1] if not self.use_img else data
    
    def predict(self, data):
        raise NotImplementedError

    def call(self, labeled_data, unlabeled_data):
        raise NotImplementedError

    def accuracy(self, pred, label):
        return tf.reduce_mean(tf.cast(pred == label, tf.float32))

    def metrics(self, pred, label):
        f1 = f1_score(label, pred, average="weighted", zero_division=0)
        acc = accuracy_score(label, pred)
        return f1, acc


class GCDMRelFramework(object):
    def __init__(
        self,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        base_category,
        novel_category,
    ) -> None:
        self.__train_dataloader = train_dataloader
        self.__val_dataloader = val_dataloader
        self.__test_dataloader = test_dataloader
        self.__train_class = base_category
        self.__val_class = base_category
        self.__test_class = novel_category
        self.__train_n_class = len(self.__train_class)
        self.__val_n_class = len(self.__val_class)
        self.__test_n_class = len(self.__test_class)

        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("[bold red]{task.fields[info]}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )

    def __get_data(self, dataloader):
        (
            s_ind,
            s_seg,
            head_idx,
            tail_idx,
            label,
            img,
        ) = next(dataloader)
        s_len = s_ind.get_shape().as_list()[-1]
        dim_img_feature = img.get_shape().as_list()[-3:]
        data = (
            tf.reshape(s_ind, shape=(-1, s_len)),
            tf.reshape(s_seg, shape=(-1, s_len)),
            tf.reshape(head_idx, shape=(-1, 1)),
            tf.reshape(tail_idx, shape=(-1, 1)),
            tf.reshape(img, shape=(-1, *dim_img_feature)),
        )
        label = tf.reshape(label, shape=(-1,))
        return (data, label)
    # daeo方法需要采用下面这个带有vae_loss的__train_model_with_batch方法，而其余的采用下面的不带vae_loss的__train_model_with_batch
    def __train_model_with_batch(self, model, optimizer, labeled_dataloader, unlabeled_dataloader):
        labeled_data, label = self.__get_data(labeled_dataloader)
        unlabeled_data, _ = self.__get_data(unlabeled_dataloader)
        with tf.GradientTape() as tape:
            vae_loss = model.train_vae_with_labeled_data(labeled_data, label)
        trainable_variables = model.vae.trainable_variables
        grads = tape.gradient(vae_loss, trainable_variables)
        optimizer.apply_gradients(zip(grads, trainable_variables))
        with tf.GradientTape() as tape:
            pred, labeled_loss, unlabeled_loss = model(labeled_data, label, unlabeled_data)
            overall_loss = labeled_loss + unlabeled_loss
        trainable_variables = [model.cluster] + model.encoder.trainable_variables + model.fc.trainable_variables
        grads = tape.gradient(overall_loss, trainable_variables)
        optimizer.apply_gradients(zip(grads, trainable_variables))
        acc = model.accuracy(pred, label)
        return overall_loss, acc

    # def __train_model_with_batch(self, model, optimizer, labeled_dataloader, unlabeled_dataloader):
    #     labeled_data, label = self.__get_data(labeled_dataloader)
    #     unlabeled_data, _ = self.__get_data(unlabeled_dataloader)
    #     with tf.GradientTape() as tape:
    #         pred, labeled_loss, unlabeled_loss = model(labeled_data, label, unlabeled_data)
    #         overall_loss = labeled_loss + unlabeled_loss
    #     grads = tape.gradient(overall_loss, model.trainable_variables)
    #     optimizer.apply_gradients(zip(grads, model.trainable_variables))
    #     acc = model.accuracy(pred, label)
    #     return overall_loss, acc

    def train(
        self,
        model,
        lr: float,
        epoch: int,
        patience: int,
        train_iter: int,
        val_iter: int,
        model_path: str,
    ):
        labeled_dataloader = self.__train_dataloader
        unlabeled_dataloader = self.__test_dataloader
        losses = []
        train_accs = []
        acc = 0.0
        n_patience = 0
        optimizer = tf.optimizers.Adam(learning_rate=lr)
        for e in range(1, epoch + 1):
            losses.clear()
            train_accs.clear()
            train_tqdm = self.progress.add_task(
                description=f"Training epoch {e}",
                total=train_iter,
                info="train_loss:--.--, train_acc:--.--%, val_acc:--.--%",
            )
            self.progress.start()
            for _ in range(train_iter):
                loss, train_acc = self.__train_model_with_batch(
                    model, optimizer, labeled_dataloader, unlabeled_dataloader
                )
                train_accs.append(train_acc)
                losses.append(loss)
                info = "train_loss: {0:2.6f}, train_acc: {1:3.2f}%, val_acc: {2:3.2f}%".format(
                    np.mean(losses), 100 * np.mean(train_accs), acc * 100
                )
                self.progress.advance(train_tqdm, advance=1)
                self.progress.update(train_tqdm, info=info)
            val_acc = self.eval(model, val_iter)
            if acc <= val_acc:
                acc = val_acc
                n_patience = 0
                self.progress.log("[bold green]Best checkpoint")
                info = "Acc {0:3.2f}".format(acc * 100)
                self.progress.log("[bold blue] Valid result: " + info)
                model.save_weights(model_path)
            else:
                n_patience += 1
                if n_patience == patience:
                    break
        self.progress.log("[bold red]Finish training " + model_path)

    def _load_model(self, model, model_path):
        if os.path.exists(model_path):
            optimizer = tf.optimizers.Adam()
            self.__train_model_with_batch(
                model, optimizer, self.__train_dataloader, self.__test_dataloader
            )
            model.load_weights(model_path, by_name=True)
        else:
            print(f"The model file [{model_path}] are not found !")

    def eval(self, model, val_iter, do_test: bool = False, model_path: str = ""):
        if model_path:
            self._load_model(model, model_path)
        if do_test:
            dataloader = self.__test_dataloader
        else:
            dataloader = self.__val_dataloader

        eval_tqdm = self.progress.add_task(
            description="Evaluating", total=val_iter, info="val_acc:--.--"
        )
        self.progress.start()

        labels, preds = [], []
        for _ in range(val_iter):
            val_data, val_label = self.__get_data(dataloader)
            pred = model.predict(val_data)
            labels.append(val_label)
            preds.append(pred)
            self.progress.advance(eval_tqdm, advance=1)
        labels = tf.concat(labels, axis=0)
        preds = tf.concat(preds, axis=0)

        preds = preds.numpy()
        labels = labels.numpy()

        D = len(self.__train_class + self.__test_class)
        base_category_gt = [i for i in range(self.__train_n_class)]
        novel_category_gt = [i for i in range(self.__train_n_class, D)]
        overall_acc, w, ind_map = self.__calc_overall_acc(D, preds, labels) 
        base_category_acc = self.__calc_category_acc(base_category_gt, w, ind_map)
        if do_test:
            novel_category_acc = self.__calc_category_acc(novel_category_gt, w, ind_map)
            return (base_category_acc, novel_category_acc, overall_acc)
        else:
            info = "val_acc:{0:3.2f}".format(base_category_acc * 100)
            self.progress.update(eval_tqdm, info=info)
            return base_category_acc

    def __calc_overall_acc(self, D, preds, labels):
        w = np.zeros((D, D), dtype=int)
        for i in range(len(preds)):
            w[preds[i], labels[i]] += 1
        ind = linear_assignment(w.max() - w)
        ind = np.vstack(ind).T
        ind_map = {j:i for i, j in ind}
        acc = sum(w[i, j] for i, j in ind)
        instances = len(preds)
        acc /= instances
        return (acc, w, ind_map)

    def __calc_category_acc(self, category_gt, w, ind_map):
        acc = 0
        instances = 0
        for i in category_gt:
            acc += w[ind_map[i], i]
            instances += sum(w[:, i])
        acc /= instances
        return acc