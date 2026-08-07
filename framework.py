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


CHECKPOINT_TIE_POLICIES = ("latest", "first")


def _checkpoint_decision(best_acc, val_acc, tie_policy, has_checkpoint):
    """Return whether to save and whether validation reset patience."""
    if tie_policy not in CHECKPOINT_TIE_POLICIES:
        raise ValueError(
            "checkpoint_tie_policy must be one of {}".format(
                CHECKPOINT_TIE_POLICIES
            )
        )
    if best_acc < val_acc:
        return True, True
    if best_acc == val_acc:
        should_save = tie_policy == "latest" or not has_checkpoint
        return should_save, True
    return False, False

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
        runtime_check_every: int = 0,
    ) -> None:
        if runtime_check_every < 0:
            raise ValueError("runtime_check_every must be greater than or equal to 0")
        self.__train_dataloader = train_dataloader
        self.__val_dataloader = val_dataloader
        self.__test_dataloader = test_dataloader
        self.__train_class = base_category
        self.__val_class = base_category
        self.__test_class = novel_category
        self.__train_n_class = len(self.__train_class)
        self.__val_n_class = len(self.__val_class)
        self.__test_n_class = len(self.__test_class)
        self.__runtime_check_every = runtime_check_every
        self.__train_step = 0

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

    def __check_batch(self, data, label, name, num_classes):
        if len(data) != 5:
            raise ValueError(f"{name} data must contain 5 tensors, got {len(data)}")

        s_ind, s_seg, head_idx, tail_idx, img = data
        for tensor_name, tensor in (
            ("sentence_indices", s_ind),
            ("sentence_segments", s_seg),
            ("head_idx", head_idx),
            ("tail_idx", tail_idx),
            ("label", label),
        ):
            tf.debugging.assert_type(
                tensor, tf.int64, message=f"{name}.{tensor_name} must be tf.int64"
            )
        tf.debugging.assert_type(
            img, tf.float32, message=f"{name}.img must be tf.float32"
        )

        tf.debugging.assert_rank(s_ind, 2, message=f"{name}.sentence_indices")
        tf.debugging.assert_rank(s_seg, 2, message=f"{name}.sentence_segments")
        tf.debugging.assert_rank(head_idx, 2, message=f"{name}.head_idx")
        tf.debugging.assert_rank(tail_idx, 2, message=f"{name}.tail_idx")
        tf.debugging.assert_rank(label, 1, message=f"{name}.label")
        tf.debugging.assert_rank(img, 4, message=f"{name}.img")

        batch_size = tf.shape(label)[0]
        tf.debugging.assert_positive(batch_size, message=f"{name} batch is empty")
        tf.debugging.assert_equal(
            tf.shape(s_ind), tf.shape(s_seg), message=f"{name} token/segment shape mismatch"
        )
        for tensor_name, tensor in (
            ("sentence_indices", s_ind),
            ("head_idx", head_idx),
            ("tail_idx", tail_idx),
            ("img", img),
        ):
            tf.debugging.assert_equal(
                tf.shape(tensor)[0],
                batch_size,
                message=f"{name}.{tensor_name} batch dimension mismatch",
            )
        tf.debugging.assert_equal(
            tf.shape(head_idx)[1], 1, message=f"{name}.head_idx must have shape [B, 1]"
        )
        tf.debugging.assert_equal(
            tf.shape(tail_idx)[1], 1, message=f"{name}.tail_idx must have shape [B, 1]"
        )
        tf.debugging.assert_equal(
            tf.shape(img)[1:],
            tf.constant([384, 384, 3], dtype=tf.int32),
            message=f"{name}.img must have shape [B, 384, 384, 3]",
        )

        seq_len = tf.shape(s_ind)[1]
        for tensor_name, tensor in (("head_idx", head_idx), ("tail_idx", tail_idx)):
            tf.debugging.assert_greater_equal(
                tensor, tf.cast(0, tensor.dtype), message=f"{name}.{tensor_name} is negative"
            )
            tf.debugging.assert_less(
                tensor,
                tf.cast(seq_len, tensor.dtype),
                message=f"{name}.{tensor_name} exceeds sequence length",
            )
        tf.debugging.assert_greater_equal(
            label, tf.cast(0, label.dtype), message=f"{name}.label is negative"
        )
        tf.debugging.assert_less(
            label,
            tf.cast(num_classes, label.dtype),
            message=f"{name}.label exceeds the configured class count",
        )
        tf.debugging.assert_all_finite(img, message=f"{name}.img contains NaN or Inf")

    def __check_vector(self, tensor, batch_size, name):
        tf.debugging.assert_rank(tensor, 1, message=f"{name} must have shape [B]")
        tf.debugging.assert_equal(
            tf.shape(tensor)[0], batch_size, message=f"{name} batch dimension mismatch"
        )
        tf.debugging.assert_all_finite(tensor, message=f"{name} contains NaN or Inf")

    def __check_gradients(self, grads, variables, name):
        if len(grads) != len(variables):
            raise ValueError(
                f"{name} gradient/variable count mismatch: {len(grads)} != {len(variables)}"
            )
        disconnected = [var.name for grad, var in zip(grads, variables) if grad is None]
        if disconnected:
            raise ValueError(f"{name} has disconnected gradients: {disconnected}")

        gradient_values = []
        for grad, variable in zip(grads, variables):
            value = grad.values if isinstance(grad, tf.IndexedSlices) else grad
            tf.debugging.assert_all_finite(
                value, message=f"{name} gradient for {variable.name} contains NaN or Inf"
            )
            gradient_values.append(value)
        grad_norm = tf.linalg.global_norm(gradient_values)
        tf.debugging.assert_all_finite(
            grad_norm, message=f"{name} global gradient norm contains NaN or Inf"
        )
        return grad_norm

    def __check_main_outputs(
        self, pred, label, labeled_loss, unlabeled_loss, overall_loss, num_classes
    ):
        batch_size = tf.shape(label)[0]
        tf.debugging.assert_rank(pred, 1, message="pred must have shape [B]")
        tf.debugging.assert_equal(
            tf.shape(pred)[0], batch_size, message="pred batch dimension mismatch"
        )
        if not pred.dtype.is_integer:
            raise TypeError(f"pred must have an integer dtype, got {pred.dtype.name}")
        tf.debugging.assert_greater_equal(
            pred, tf.cast(0, pred.dtype), message="pred contains a negative class index"
        )
        tf.debugging.assert_less(
            pred,
            tf.cast(num_classes, pred.dtype),
            message="pred exceeds the configured class count",
        )
        self.__check_vector(labeled_loss, batch_size, "labeled_loss")
        tf.debugging.assert_rank(unlabeled_loss, 0, message="unlabeled_loss must be scalar")
        tf.debugging.assert_all_finite(
            unlabeled_loss, message="unlabeled_loss contains NaN or Inf"
        )
        self.__check_vector(overall_loss, batch_size, "overall_loss")

    def __check_module_b_diagnostics(self, model):
        if not hasattr(model.encoder, "module_b_diagnostics"):
            return

        diagnostics = model.encoder.module_b_diagnostics()
        required = (
            "text_curvature",
            "image_curvature",
            "text_weight",
            "image_weight",
            "attention_entropy",
            "residual_ratio",
            "manifold_error",
        )
        missing = [name for name in required if name not in diagnostics]
        if missing:
            raise ValueError(f"module B diagnostics are missing: {missing}")
        for name in required:
            value = diagnostics[name]
            tf.debugging.assert_rank(value, 0, message=f"module B {name} must be scalar")
            tf.debugging.assert_all_finite(
                value, message=f"module B {name} contains NaN or Inf"
            )

        tf.debugging.assert_positive(
            diagnostics["text_curvature"], message="module B text curvature is not positive"
        )
        tf.debugging.assert_positive(
            diagnostics["image_curvature"], message="module B image curvature is not positive"
        )
        for name in ("text_weight", "image_weight"):
            tf.debugging.assert_greater_equal(
                diagnostics[name], 0.0, message=f"module B {name} is negative"
            )
            tf.debugging.assert_less_equal(
                diagnostics[name], 1.0, message=f"module B {name} exceeds one"
            )
        tf.debugging.assert_near(
            diagnostics["text_weight"] + diagnostics["image_weight"],
            tf.constant(1.0, dtype=tf.float32),
            atol=1e-5,
            message="module B mean modality weights do not sum to one",
        )
        tf.debugging.assert_greater_equal(
            diagnostics["attention_entropy"],
            0.0,
            message="module B attention entropy is negative",
        )
        tf.debugging.assert_less_equal(
            diagnostics["attention_entropy"],
            tf.math.log(tf.constant(2.0, dtype=tf.float32)) + 1e-5,
            message="module B attention entropy exceeds log(2)",
        )
        tf.debugging.assert_greater_equal(
            diagnostics["residual_ratio"],
            0.0,
            message="module B residual ratio is negative",
        )
        tf.debugging.assert_less_equal(
            diagnostics["manifold_error"],
            1e-3,
            message="module B Lorentz constraint error is too large",
        )
        tf.print(
            "[module-b]",
            "text_curvature=", diagnostics["text_curvature"],
            "image_curvature=", diagnostics["image_curvature"],
            "text_weight=", diagnostics["text_weight"],
            "image_weight=", diagnostics["image_weight"],
            "attention_entropy=", diagnostics["attention_entropy"],
            "residual_ratio=", diagnostics["residual_ratio"],
            "manifold_error=", diagnostics["manifold_error"],
        )

    def __check_module_c_diagnostics(self, model):
        if not hasattr(model.encoder, "module_c_diagnostics"):
            return

        diagnostics = model.encoder.module_c_diagnostics()
        required = (
            "attribute_weight",
            "text_weight",
            "gate_std",
            "saturation",
            "brightness",
            "texture_proxy",
            "contrast",
            "residual_ratio",
        )
        missing = [name for name in required if name not in diagnostics]
        if missing:
            raise ValueError(f"module C diagnostics are missing: {missing}")
        for name in required:
            value = diagnostics[name]
            tf.debugging.assert_rank(value, 0, message=f"module C {name} must be scalar")
            tf.debugging.assert_all_finite(
                value, message=f"module C {name} contains NaN or Inf"
            )

        for name in ("attribute_weight", "text_weight"):
            tf.debugging.assert_greater_equal(
                diagnostics[name], 0.0, message=f"module C {name} is negative"
            )
            tf.debugging.assert_less_equal(
                diagnostics[name], 1.0, message=f"module C {name} exceeds one"
            )
        tf.debugging.assert_near(
            diagnostics["attribute_weight"] + diagnostics["text_weight"],
            tf.constant(1.0, dtype=tf.float32),
            atol=1e-5,
            message="module C mean gate weights do not sum to one",
        )
        tf.debugging.assert_greater_equal(
            diagnostics["gate_std"], 0.0, message="module C gate std is negative"
        )
        tf.debugging.assert_less_equal(
            diagnostics["gate_std"], 0.5 + 1e-5, message="module C gate std is invalid"
        )
        for name in ("saturation", "brightness", "texture_proxy", "contrast"):
            tf.debugging.assert_greater_equal(
                diagnostics[name], 0.0, message=f"module C {name} is negative"
            )
            tf.debugging.assert_less_equal(
                diagnostics[name], 1.0, message=f"module C {name} exceeds one"
            )
        tf.debugging.assert_greater_equal(
            diagnostics["residual_ratio"],
            0.0,
            message="module C residual ratio is negative",
        )
        tf.print(
            "[module-c]",
            "attribute_weight=", diagnostics["attribute_weight"],
            "text_weight=", diagnostics["text_weight"],
            "gate_std=", diagnostics["gate_std"],
            "saturation=", diagnostics["saturation"],
            "brightness=", diagnostics["brightness"],
            "texture_proxy=", diagnostics["texture_proxy"],
            "contrast=", diagnostics["contrast"],
            "residual_ratio=", diagnostics["residual_ratio"],
        )
    # daeo方法需要采用下面这个带有vae_loss的__train_model_with_batch方法，而其余的采用下面的不带vae_loss的__train_model_with_batch
    def __train_model_with_batch(
        self,
        model,
        optimizer,
        labeled_dataloader,
        unlabeled_dataloader,
        force_runtime_checks=False,
        labeled_batch=None,
        unlabeled_batch=None,
    ):
        if labeled_batch is None:
            labeled_data, label = self.__get_data(labeled_dataloader)
        else:
            labeled_data, label = labeled_batch
        if unlabeled_batch is None:
            unlabeled_data, unlabeled_label = self.__get_data(unlabeled_dataloader)
        else:
            unlabeled_data, unlabeled_label = unlabeled_batch
        step = self.__train_step + 1
        should_check = force_runtime_checks or (
            self.__runtime_check_every > 0
            and (step - 1) % self.__runtime_check_every == 0
        )
        if should_check:
            self.__check_batch(labeled_data, label, "labeled", model.K)
            self.__check_batch(unlabeled_data, unlabeled_label, "unlabeled", model.K)
        with tf.GradientTape() as tape:
            vae_loss = model.train_vae_with_labeled_data(labeled_data, label)
        trainable_variables = model.vae.trainable_variables
        grads = tape.gradient(vae_loss, trainable_variables)
        if should_check:
            self.__check_vector(vae_loss, tf.shape(label)[0], "vae_loss")
            vae_grad_norm = self.__check_gradients(grads, trainable_variables, "vae")
        optimizer.apply_gradients(zip(grads, trainable_variables))
        with tf.GradientTape() as tape:
            pred, labeled_loss, unlabeled_loss = model(labeled_data, label, unlabeled_data)
            overall_loss = labeled_loss + unlabeled_loss
        trainable_variables = [model.cluster] + model.encoder.trainable_variables + model.fc.trainable_variables
        grads = tape.gradient(overall_loss, trainable_variables)
        if should_check:
            self.__check_main_outputs(
                pred, label, labeled_loss, unlabeled_loss, overall_loss, model.K
            )
            main_grad_norm = self.__check_gradients(grads, trainable_variables, "main")
        optimizer.apply_gradients(zip(grads, trainable_variables))
        acc = model.accuracy(pred, label)
        if should_check:
            tf.debugging.assert_all_finite(acc, message="accuracy contains NaN or Inf")
            self.__check_module_b_diagnostics(model)
            self.__check_module_c_diagnostics(model)
            tf.print(
                "[runtime-check]",
                "step=", step,
                "labeled_input=", tf.shape(labeled_data[0]),
                "unlabeled_input=", tf.shape(unlabeled_data[0]),
                "image=", tf.shape(labeled_data[-1]),
                "pred=", tf.shape(pred),
                "vae_loss_mean=", tf.reduce_mean(vae_loss),
                "labeled_loss_mean=", tf.reduce_mean(labeled_loss),
                "unlabeled_loss=", unlabeled_loss,
                "overall_loss_mean=", tf.reduce_mean(overall_loss),
                "vae_grad_norm=", vae_grad_norm,
                "main_grad_norm=", main_grad_norm,
            )
        self.__train_step = step
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

    def smoke_test(self, model, lr: float):
        optimizer = tf.optimizers.Adam(learning_rate=lr)
        loss, accuracy = self.__train_model_with_batch(
            model,
            optimizer,
            self.__train_dataloader,
            self.__test_dataloader,
            force_runtime_checks=True,
        )
        return {
            "loss": float(tf.reduce_mean(loss).numpy()),
            "accuracy": float(accuracy.numpy()),
        }

    def overfit_test(self, model, lr: float, steps: int = 20):
        if steps <= 0:
            raise ValueError("overfit steps must be positive")
        optimizer = tf.optimizers.Adam(learning_rate=lr)
        labeled_batch = self.__get_data(self.__train_dataloader)
        unlabeled_batch = self.__get_data(self.__test_dataloader)
        history = []
        for index in range(steps):
            loss, accuracy = self.__train_model_with_batch(
                model,
                optimizer,
                self.__train_dataloader,
                self.__test_dataloader,
                force_runtime_checks=index in (0, steps - 1),
                labeled_batch=labeled_batch,
                unlabeled_batch=unlabeled_batch,
            )
            mean_loss = float(tf.reduce_mean(loss).numpy())
            accuracy_value = float(accuracy.numpy())
            history.append((mean_loss, accuracy_value))
            print(
                "[overfit] step={0}/{1} loss={2:.6f} accuracy={3:.2f}%".format(
                    index + 1, steps, mean_loss, accuracy_value * 100
                )
            )
        return {
            "initial_loss": history[0][0],
            "final_loss": history[-1][0],
            "initial_accuracy": history[0][1],
            "final_accuracy": history[-1][1],
        }

    def train(
        self,
        model,
        lr: float,
        epoch: int,
        patience: int,
        train_iter: int,
        val_iter: int,
        model_path: str,
        checkpoint_tie_policy: str = "latest",
    ):
        if checkpoint_tie_policy not in CHECKPOINT_TIE_POLICIES:
            raise ValueError(
                "checkpoint_tie_policy must be one of {}".format(
                    CHECKPOINT_TIE_POLICIES
                )
            )
        labeled_dataloader = self.__train_dataloader
        unlabeled_dataloader = self.__test_dataloader
        losses = []
        train_accs = []
        acc = 0.0
        n_patience = 0
        has_checkpoint = False
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
            should_save, reset_patience = _checkpoint_decision(
                acc,
                val_acc,
                checkpoint_tie_policy,
                has_checkpoint,
            )
            if reset_patience:
                acc = val_acc
                n_patience = 0
                if should_save:
                    self.progress.log("[bold green]Best checkpoint")
                    info = "Acc {0:3.2f}".format(acc * 100)
                    self.progress.log("[bold blue] Valid result: " + info)
                    model.save_weights(model_path)
                    has_checkpoint = True
                else:
                    self.progress.log(
                        "[bold yellow]Validation tie: keeping earlier checkpoint"
                    )
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
