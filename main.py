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
    parser.add_argument("--model_path", type=str, default="./weights")
    parser.add_argument("--fine_tune", type=int, choices=[0, 1], default=1)
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train")
    parser.add_argument("--model", type=str, choices=["kmeans", "rankstats", "uno", "gcd", "simgcd", "daeo", "sae"])
    args = parser.parse_args()

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
    if args.mode == "train":
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
    )

    if args.mode == "train":
        framework.train(
            model,
            args.lr,
            args.epoch,
            args.patience,
            train_iter,
            val_iter,
            model_path,
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