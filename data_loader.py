import os
import re
import codecs
import json
import hashlib
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from threading import Thread
from copy import deepcopy
from vit_keras import vit
from typing import List, Dict
from tensorflow.data import Dataset
from keras_bert.tokenizer import Tokenizer
from tensorflow.keras.preprocessing import image
from keras_bert import TOKEN_CLS, TOKEN_MASK, TOKEN_SEP


class MRelSample(Thread):
    def __init__(self, **kwargs):
        Thread.__init__(self)

        # 先初始化，避免线程加载失败后没有 __image 属性
        self.__image = None

        if "sample" in kwargs:
            sample = kwargs["sample"]
            self.__token = sample.token
            self.__image = sample.image
            self.__relation = sample.relation
            self.__head_ent = sample.head_entity
            self.__tail_ent = sample.tail_entity
        else:
            dict_data = kwargs["dict_data"]

            self.__head_ent = dict_data["h"]
            self.__tail_ent = dict_data["t"]
            self.__token = dict_data["tokens"]
            self.__img_id = dict_data["img_id"]
            self.__relation = dict_data["relation"]
            self.__image_data_path = kwargs["image_data_path"]

    def run(self):
        # 你的图片文件是 000001.jpg，所以这里必须加 .jpg
        img_path = os.path.join(self.__image_data_path, self.__img_id + ".jpg")

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        self.__image = self.__load_image(img_path)

    def __load_image(self, fname: str, image_size: int = 384):
        img = image.load_img(fname, target_size=(image_size, image_size))
        x = image.img_to_array(img)
        x = vit.preprocess_inputs(x).reshape(image_size, image_size, 3)
        return x

    @property
    def image(self):
        if self.__image is None:
            raise ValueError("Image is None. 图片可能没有成功加载，请检查 img_id 和 image 路径。")
        return self.__image

    @property
    def token(self):
        return self.__token

    @property
    def head_entity(self):
        return self.__head_ent

    @property
    def tail_entity(self):
        return self.__tail_ent

    @property
    def relation(self):
        return self.__relation


class MRelDataset(object):
    def __init__(self, data_path: str):
        self.__samples = self.__collect_data(data_path)
        self.__dict_rel2samples = self.__parse_relation_dict(self.__samples)

    @property
    def rel2sampleIdxs(self):
        return self.__dict_rel2samples

    @property
    def relations(self):
        return list(self.__dict_rel2samples.keys())

    def idx2samples(self, idxs: List[int]) -> List[MRelSample]:
        samples = []
        for i in idxs:
            samples.append(MRelSample(sample=self.__samples[i]))
        return samples

    def __parse_relation_dict(self, samples: List[MRelSample]) -> Dict:
        dict_rel2samples = {}
        for i, sample in enumerate(samples):
            relation = sample.relation
            if relation not in dict_rel2samples:
                dict_rel2samples[relation] = []
            dict_rel2samples[relation].append(i)
        return dict_rel2samples

    def __collect_data(self, data_path: str) -> List[MRelSample]:
        samples = []
        for dtype in ["train", "test"]:
            with open(f"{data_path}/{dtype}.txt", "r") as fr:
                for line in tqdm(fr.readlines(), ncols=80, ascii=True):
                    sample = MRelSample(dict_data=eval(line), image_data_path=f"{data_path}/image")
                    sample.start()
                    samples.append(sample)
                for sample in samples:
                    sample.join()
        return samples

def split_dataset(dataset, train_types, test_types):
    train_dataset = {k:[] for k in train_types}
    val_dataset = {k:[] for k in train_types}
    test_dataset = {k:[] for k in train_types + test_types}
    rel2sampleIdxs = dataset.rel2sampleIdxs

    for train_type in train_types:
        train_type_sampleIdxs = rel2sampleIdxs[train_type]
        n_seen_train_val_sample = int(len(train_type_sampleIdxs) * 0.5)
        n_seen_test_sample = len(train_type_sampleIdxs) - n_seen_train_val_sample
        n_seen_train_sample = int(n_seen_train_val_sample * 0.8)
        n_seen_val_sample = n_seen_train_val_sample - n_seen_train_sample

        np.random.shuffle(train_type_sampleIdxs)

        seen_train_val_sampleIdxs =  train_type_sampleIdxs[:n_seen_train_val_sample]
        seen_test_sampleIdxs = train_type_sampleIdxs[-n_seen_test_sample:]
        assert len(seen_train_val_sampleIdxs) + len(seen_test_sampleIdxs) == len(train_type_sampleIdxs)

        seen_train_sampleIdxs = seen_train_val_sampleIdxs[:n_seen_train_sample]
        seen_val_sampleIdxs = seen_train_val_sampleIdxs[-n_seen_val_sample:]
        assert len(seen_train_sampleIdxs) + len(seen_val_sampleIdxs) == len(seen_train_val_sampleIdxs)

        train_dataset[train_type] += dataset.idx2samples(seen_train_sampleIdxs)
        val_dataset[train_type] += dataset.idx2samples(seen_val_sampleIdxs)
        test_dataset[train_type] += dataset.idx2samples(seen_test_sampleIdxs)
    
    for test_type in test_types:
        test_dataset[test_type] += dataset.idx2samples(rel2sampleIdxs[test_type])

    return train_dataset, val_dataset, test_dataset

class TextProcessor(object):
    def __init__(
        self, bert_path: str, max_length: int = 128, mask_entity: bool = False
    ):
        self.__mask_entity = mask_entity
        self.__max_length = max_length
        self.__tokenizer = self.__load_tokenizer(bert_path)

    def tokenize_sentence(self, sentence: str, max_len=16):
        indices, segments = self.__tokenizer.encode(sentence, max_len=max_len)
        return (indices, segments)

    def tokenize_sample(self, raw_tokens: List, pos_head: List, pos_tail: List):
        pos_head[-1] -= 1
        pos_tail[-1] -= 1
        tokens = [TOKEN_CLS]
        cur_pos = 0
        pos1_in_index = 1
        pos2_in_index = 1
        for token in raw_tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                tokens.append("[unused1]")
                pos1_in_index = len(tokens)
            if cur_pos == pos_tail[0]:
                tokens.append("[unused2]")
                pos2_in_index = len(tokens)
            if self.__mask_entity and (
                (pos_head[0] <= cur_pos and cur_pos <= pos_head[-1])
                or (pos_tail[0] <= cur_pos and cur_pos <= pos_tail[-1])
            ):
                tokens += ["[unused5]"]
            else:
                tokens += self.__tokenizer.tokenize(token)[1:-1]
            if cur_pos == pos_head[-1]:
                tokens.append("[unused3]")
            if cur_pos == pos_tail[-1]:
                tokens.append("[unused4]")
            cur_pos += 1
        tokens.append(TOKEN_SEP)
        indices = self.__tokenizer._convert_tokens_to_ids(tokens)

        # padding
        while len(indices) < self.__max_length:
            indices.append(0)
        indices = indices[: self.__max_length]
        segments = np.zeros_like(indices).tolist()

        pos1_in_index = min(self.__max_length, pos1_in_index)
        pos2_in_index = min(self.__max_length, pos2_in_index)

        return indices, segments, pos1_in_index - 1, pos2_in_index - 1

    def __load_tokenizer(self, bert_path):
        token_dict = {}
        with codecs.open(f"{bert_path}/vocab.txt", "r", "utf8") as reader:
            for line in reader:
                token = line.strip()
                token_dict[token] = len(token_dict)
        return Tokenizer(token_dict)


class GCDMRelDataset(object):
    def __init__(
        self,
        batch_size: int,
        classes: List[str],
        dataset: Dict,
        textProcessor: TextProcessor,
        shuffle: bool = False,
        balanced_sampling: bool = False,
    ):
        self.__classes = classes
        self.__batch_size = batch_size
        self.__textProcessor = textProcessor
        (
            self.__dataset,
            self.__sample_index,
            self.__total_step,
        ) = self.__generate_data_index(dataset)
        self.__cur_step = 0
        self.__shuffle = shuffle
        self.__balanced_sampling = balanced_sampling

    @property
    def step(self):
        if self.__batch_size == 1:
            return self.__total_step
        else:
            return self.__total_step // self.__batch_size + 1

    def _generate_data(self):
        dict_sample_index = {}
        if self.__balanced_sampling:
            for i, j in self.__sample_index:
                if i not in dict_sample_index:
                    dict_sample_index[i] = []
                dict_sample_index[i].append((i, j))
        else:
            sample_index = self.__sample_index
            if self.__shuffle:
                np.random.shuffle(sample_index)
        del self.__sample_index
        while True:
            if self.__balanced_sampling:
                target_class_idx = dict_sample_index[
                    self.__cur_step % len(self.__classes)
                ]
                idx = target_class_idx[
                    np.random.choice(np.arange(len(target_class_idx)))
                ]
            else:
                idx = sample_index[self.__cur_step]
            packed_data = self.__getitem(idx)
            self.__cur_step += 1
            if self.__cur_step == self.__total_step:
                self.__cur_step = 0
                if self.__shuffle:
                    np.random.shuffle(sample_index)
            yield packed_data["sentence_indices"], packed_data[
                "sentence_segments"
            ], packed_data["head_in_index"], packed_data["tail_in_index"], packed_data[
                "label"
            ], packed_data[
                "img"
            ]

    def __generate_data_index(self, original_dataset):
        dataset = {}
        total_step = 0
        sample_index = []
        for i, class_name in enumerate(self.__classes):
            if class_name not in dataset:
                dataset[class_name] = []
            samples = original_dataset[class_name]
            for j, sample in enumerate(samples):
                dataset[class_name].append(sample)
                total_step += 1
                sample_index.append((i, j))
        return dataset, sample_index, total_step

    def __getitem(self, sample_index):
        i, j = sample_index
        target_type = self.__classes[i]
        packed_data = {
            "sentence_indices": [],
            "sentence_segments": [],
            "head_in_index": [],
            "tail_in_index": [],
            "img": [],
            "label": [],
        }
        sample = self.__dataset[target_type][j]
        token = sample.token
        image = sample.image
        pos_head = deepcopy(sample.head_entity["pos"])
        pos_tail = deepcopy(sample.tail_entity["pos"])
        (
            sentence_indices,
            sentence_segments,
            head_in_index,
            tail_in_index,
        ) = self.__textProcessor.tokenize_sample(token, pos_head, pos_tail)
        packed_data["sentence_indices"].append(sentence_indices)
        packed_data["sentence_segments"].append(sentence_segments)
        packed_data["head_in_index"].append(head_in_index)
        packed_data["tail_in_index"].append(tail_in_index)
        packed_data["img"].append(image)
        packed_data["label"].append(self.__classes.index(target_type))
        return packed_data


def get_loader(
    classes,
    dataset,
    textProcessor,
    batch_size: int,
    shuffle: bool = False,
    balanced_sampling: bool = False,
):
    gcd_dataset = GCDMRelDataset(
        batch_size, classes, dataset, textProcessor, shuffle, balanced_sampling,
    )
    dataloader = Dataset.from_generator(
        gcd_dataset._generate_data,
        tuple([tf.int64] * 5 + [tf.float32]),
        tuple(
            [tf.TensorShape([None, None])] * 2
            + [tf.TensorShape([None])] * 3 + [tf.TensorShape([None] * 4)]
        ),
    )
    dataloader = dataloader.batch(batch_size)
    return iter(dataloader), gcd_dataset.step


if __name__ == "__main__":
    import gc
    from utils import split_types
    batch_size = 4
    dataset = MRelDataset("../2024-COLING-MCIL-code/dataset/MRE")
    textProcessor = TextProcessor("../ZS-MNET/pretrain/cased_L-12_H-768_A-12")
    relations = dataset.relations

    # parsed_relations = []

    # for relation in relations:
    #     if "/" in relation:
    #         relation = relation.split("/")[-1]
    #     parsed_relations.append(relation)
    
    # parsed_relations = set(parsed_relations)
    # relations = set(relations)
    # print(parsed_relations)
    # print(relations)

    # train_types, val_types, test_types = split_types(dataset.relations)

    # print("train:", train_types)
    # print("valid:", val_types)
    # print("test:", test_types)

    # train_dataloader, train_iter = get_loader(
    #     train_types, dataset, textProcessor, batch_size, balanced_sampling=True
    # )
    # val_dataloader, val_iter = get_loader(val_types, dataset, textProcessor, batch_size)
    # test_dataloader, test_iter = get_loader(
    #     test_types, dataset, textProcessor, batch_size=batch_size, shuffle=False
    # )

    # del dataset
    # gc.collect() 
