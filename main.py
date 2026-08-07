import os
import gc
import random
import pprint
import argparse
import numpy as np
from model import *
import tensorflow as tf
from utils import split_types
from sentence_encoder import *
from multimodal_encoder import *
from framework import GCDMRelFramework
from data_loader import get_loader, MRelDataset, TextProcessor, split_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_path", type=str, default="/home/zhoubaohang/FewRel")
    parser.add_argument(
        "--bert_path", type=str, default="../../official-pretrained-models/cased_L-12_H-768_A-12"
    )
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epoch", type=int, default=25)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--checkpoint_tie_policy",
        type=str,
        choices=["latest", "first"],
        default="latest",
        help=(
            "Checkpoint to keep when validation accuracy ties: latest preserves "
            "the original behavior; first keeps the earlier checkpoint from "
            "the current training run."
        ),
    )
    parser.add_argument("--model_path", type=str, default="./weights")
    parser.add_argument("--fine_tune", type=int, choices=[0, 1], default=1)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test", "smoke", "overfit"],
        default="train",
    )
    parser.add_argument(
        "--overfit_steps",
        type=int,
        default=20,
        help="Number of repeated updates on one cached batch in --mode overfit.",
    )
    parser.add_argument(
        "--runtime_check_every",
        type=int,
        default=0,
        help="Run shape/finite/gradient checks every N training steps; 0 disables them.",
    )
    parser.add_argument(
        "--use_module_b",
        type=int,
        choices=[0, 1],
        default=0,
        help="Enable the adapted curvature-aware fusion module; disabled by default.",
    )
    parser.add_argument("--module_b_dim", type=int, default=128)
    parser.add_argument("--module_b_residual_scale", type=float, default=0.05)
    parser.add_argument("--module_b_curvature_init", type=float, default=1.0)
    parser.add_argument("--module_b_curvature_min", type=float, default=0.05)
    parser.add_argument("--module_b_curvature_max", type=float, default=5.0)
    parser.add_argument("--module_b_prior_strength", type=float, default=0.1)
    parser.add_argument(
        "--module_b_temperature",
        type=float,
        default=0.0,
        help="Positive initial temperature; 0 uses sqrt(module_b_dim).",
    )
    parser.add_argument("--module_b_tangent_scale", type=float, default=1.0)
    parser.add_argument("--module_b_tangent_norm_max", type=float, default=2.0)
    parser.add_argument(
        "--use_module_c",
        type=int,
        choices=[0, 1],
        default=0,
        help="Enable the adapted TEDFusion attribute-text gate; disabled by default.",
    )
    parser.add_argument("--module_c_gate_dim", type=int, default=64)
    parser.add_argument("--module_c_residual_scale", type=float, default=0.01)
    parser.add_argument("--model", type=str, choices=["kmeans", "rankstats", "uno", "gcd", "simgcd", "daeo", "sae"])
    args = parser.parse_args()
    if args.runtime_check_every < 0:
        parser.error("--runtime_check_every must be greater than or equal to 0")
    if args.mode in ("smoke", "overfit") and args.model != "daeo":
        parser.error("--mode smoke/overfit currently supports only --model daeo")
    if args.overfit_steps <= 0:
        parser.error("--overfit_steps must be positive")
    if args.use_module_b and args.model != "daeo":
        parser.error("--use_module_b currently supports only --model daeo")
    if args.use_module_c and args.model != "daeo":
        parser.error("--use_module_c currently supports only --model daeo")
    if args.use_module_b and args.use_module_c:
        parser.error("modules B and C cannot be combined before stage 8")
    if args.module_b_dim <= 0:
        parser.error("--module_b_dim must be positive")
    if args.module_b_residual_scale < 0:
        parser.error("--module_b_residual_scale must be non-negative")
    if not 0 < args.module_b_curvature_min < args.module_b_curvature_max:
        parser.error("module B curvature bounds must satisfy 0 < min < max")
    if not (
        args.module_b_curvature_min
        <= args.module_b_curvature_init
        <= args.module_b_curvature_max
    ):
        parser.error("--module_b_curvature_init must lie inside its bounds")
    if args.module_b_prior_strength < 0:
        parser.error("--module_b_prior_strength must be non-negative")
    if args.module_b_temperature < 0:
        parser.error("--module_b_temperature must be non-negative")
    if args.module_b_tangent_scale <= 0 or args.module_b_tangent_norm_max <= 0:
        parser.error("module B tangent scale and norm limit must be positive")
    if args.module_c_gate_dim <= 0:
        parser.error("--module_c_gate_dim must be positive")
    if args.module_c_residual_scale < 0:
        parser.error("--module_c_residual_scale must be non-negative")

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    gpus = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(gpus[0], True)

    batch_size = args.batch_size
    max_seq_len = args.max_seq_len

    model_name = f"{args.model}-{args.seed}"
    model_path = f"{args.model_path}/{model_name}.h5"

    dataset = MRelDataset(args.data_path)
    textProcessor = TextProcessor(args.bert_path, max_seq_len)

    base_category, novel_category = split_types(dataset.relations)
    K = len(base_category + novel_category)

    print("Base:", base_category)
    print("Novel:", novel_category)

    train_dataset, val_dataset, test_dataset = split_dataset(dataset, base_category, novel_category)

    train_dataloader, train_iter = get_loader(
        base_category, train_dataset, textProcessor, batch_size, balanced_sampling=True
    )
    val_dataloader, val_iter = get_loader(base_category, val_dataset, textProcessor, batch_size)
    if args.mode in ("train", "smoke", "overfit"):
        test_dataloader, test_iter = get_loader(
            base_category + novel_category, test_dataset, textProcessor, batch_size, shuffle=True
        )
    elif args.mode == "test":
        test_dataloader, test_iter = get_loader(
            base_category + novel_category, test_dataset, textProcessor, batch_size=1
        )

    del dataset
    gc.collect()

    sentence_encoder = BERTSentenceEncoder(args.bert_path, fine_tune=bool(args.fine_tune))
    multimodal_encoder = SimpleMultimodalEncoder(sentence_encoder)
    if args.use_module_b:
        from modular_multimodal_encoder import ModuleBMultimodalEncoder

        temperature_init = (
            None if args.module_b_temperature == 0 else args.module_b_temperature
        )
        multimodal_encoder = ModuleBMultimodalEncoder(
            multimodal_encoder,
            fusion_dim=args.module_b_dim,
            residual_scale=args.module_b_residual_scale,
            curvature_init=args.module_b_curvature_init,
            curvature_min=args.module_b_curvature_min,
            curvature_max=args.module_b_curvature_max,
            prior_strength_init=args.module_b_prior_strength,
            temperature_init=temperature_init,
            tangent_scale=args.module_b_tangent_scale,
            tangent_norm_max=args.module_b_tangent_norm_max,
            initializer_seed=args.seed + 10000,
        )
        print(
            "Module B enabled:",
            {
                "fusion_dim": args.module_b_dim,
                "residual_scale": args.module_b_residual_scale,
                "curvature_init": args.module_b_curvature_init,
                "curvature_bounds": (
                    args.module_b_curvature_min,
                    args.module_b_curvature_max,
                ),
                "prior_strength": args.module_b_prior_strength,
                "temperature": temperature_init or "sqrt(fusion_dim)",
                "tangent_scale": args.module_b_tangent_scale,
                "tangent_norm_max": args.module_b_tangent_norm_max,
                "initializer_seed": args.seed + 10000,
            },
        )
    elif args.use_module_c:
        from module_c_multimodal_encoder import ModuleCMultimodalEncoder

        multimodal_encoder = ModuleCMultimodalEncoder(
            multimodal_encoder,
            gate_dim=args.module_c_gate_dim,
            residual_scale=args.module_c_residual_scale,
            initializer_seed=args.seed + 20000,
        )
        print(
            "Module C enabled:",
            {
                "gate_dim": args.module_c_gate_dim,
                "residual_scale": args.module_c_residual_scale,
                "attributes": [
                    "saturation",
                    "brightness",
                    "texture_proxy",
                    "contrast",
                ],
                "initializer_seed": args.seed + 20000,
            },
        )
    if args.model == "kmeans":
        model = KMeans(multimodal_encoder, K, 768 * 4, use_img=True)
    elif args.model == "rankstats":
        model = RankStats(multimodal_encoder, len(base_category), K, use_img=True)
    elif args.model == "uno":
        model = UNO(multimodal_encoder, K, use_img=True)
    elif args.model == "gcd":
        model = GCD(multimodal_encoder, K, 768 * 4, use_img=True)
    elif args.model == "simgcd":
        model = SimGCD(multimodal_encoder, K, 768 * 4, use_img=True)
    elif args.model == "daeo":
        model = DAEO(multimodal_encoder, K, 768 * 4, use_img=True)
    elif args.model == "sae":
        model = SAE(multimodal_encoder, K, 768 * 4, use_img=True)

    framework = GCDMRelFramework(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        base_category=base_category,
        novel_category=novel_category,
        runtime_check_every=args.runtime_check_every,
    )

    if args.mode == "smoke":
        smoke_metrics = framework.smoke_test(model, args.lr)
        print(
            "Smoke test passed: loss={0:.6f}, train_acc={1:.2f}%".format(
                smoke_metrics["loss"], smoke_metrics["accuracy"] * 100
            )
        )
    elif args.mode == "overfit":
        overfit_metrics = framework.overfit_test(model, args.lr, args.overfit_steps)
        print(
            "Overfit test finished: "
            "initial_loss={0:.6f}, final_loss={1:.6f}, "
            "initial_acc={2:.2f}%, final_acc={3:.2f}%".format(
                overfit_metrics["initial_loss"],
                overfit_metrics["final_loss"],
                overfit_metrics["initial_accuracy"] * 100,
                overfit_metrics["final_accuracy"] * 100,
            )
        )
    elif args.mode == "train":
        framework.train(
            model,
            args.lr,
            args.epoch,
            args.patience,
            train_iter,
            val_iter,
            model_path,
            args.checkpoint_tie_policy,
        )
    elif args.mode == "test":
        base_acc, novel_acc, overall_acc = framework.eval(
            model, test_iter, do_test=True, model_path=model_path
        )
        print(f"Base:{base_acc} Novel:{novel_acc} Overall:{overall_acc}")
        test_metrics = {"Base":base_acc, "Novel":novel_acc, "Overall":overall_acc}
        with open(f"{args.model_path}/{model_name}.txt", "w") as fw:
            pprint.pprint(test_metrics, stream=fw)
        os.remove(model_path)

if __name__ == "__main__":
    main()
