import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Tuple
from collections import Counter
from torch.utils.data import Dataset
from dataclasses import dataclass, field

def pad_1d(tensor: torch.Tensor, length: int, pad_value: int = 0):
    if tensor.size(0) >= length:
        return tensor[:length]
    else:
        return torch.cat([tensor, torch.full((length - tensor.size(0),), pad_value, dtype=tensor.dtype)], dim=0)

def pad_2d(tensor: torch.Tensor, lengths: Tuple[int, int], pad_value: int = 0):
    if tensor.size(0) >= lengths[0]:
        tensor = tensor[:lengths[0], :]
    if tensor.size(1) >= lengths[1]:
        tensor = tensor[:, :lengths[1]]

    if tensor.size(0) < lengths[0]:
        tensor = torch.cat([tensor, torch.full((lengths[0] - tensor.size(0), tensor.size(1)), pad_value, dtype=tensor.dtype)], dim=0)
    if tensor.size(1) < lengths[1]:
        tensor = torch.cat([tensor, torch.full((tensor.size(0), lengths[1] - tensor.size(1)), pad_value, dtype=tensor.dtype)], dim=1)
    
    return tensor

def collate_fn(batch):
    max_word_length = max(sentence["pos_tag_ids"].size(0) for sentence in batch)
    max_subword_length = max(sentence["subword_ids"].size(0) for sentence in batch)

    return {
        "subword_ids": torch.stack(
            [pad_1d(sentence["subword_ids"], max_subword_length, 0) for sentence in batch],
            dim=0
        ),
        "pos_tag_ids": torch.stack(
            [pad_1d(sentence["pos_tag_ids"], max_word_length, -1) for sentence in batch],
            dim=0
        ),
        "dependencies": torch.stack(
            [pad_1d(sentence["dependencies"], max_word_length, -1) for sentence in batch],
            dim=0
        ),
        "dep_relation_ids": torch.stack(
            [pad_1d(sentence["dep_relation_ids"], max_word_length, -1) for sentence in batch],
            dim=0
        ),
        "subword_to_word_map": torch.stack(
            [pad_1d(sentence["subword_to_word_map"], max_subword_length, -1) for sentence in batch],
            dim=0
        ),
        "word_to_subword_map": torch.stack(
            [pad_1d(sentence["word_to_subword_map"], max_word_length, -1) for sentence in batch],
            dim=0
        ),
        "attention_mask": torch.stack(
            [pad_1d(torch.ones(sentence["subword_ids"].size(0), dtype=torch.bool), max_subword_length, False) for sentence in batch],
            dim=0
        )
    }

@dataclass
class Sentence:
    words: List[str] = field(default_factory=list) 
    subwords: List[str] = field(default_factory=list) 
    subword_ids: torch.LongTensor = None
    subword_to_word_map: List[int] = field(default_factory=list)
    word_to_subword_map: List[int] = field(default_factory=list)
    pos_tags: List[str] = field(default_factory=list) 
    pos_tag_ids: torch.LongTensor = None
    dependencies: List[int] = field(default_factory=list)
    dep_relations: List[str] = field(default_factory=list)

class ConlluDataset(Dataset):
    def __init__(self, path: str, tokenizer: AutoTokenizer, pos_ids_to_str: List[str] = None, dep_ids_to_str: List[str] = None, verbose=True):
        self.sentences = []
        sentence = Sentence()
        space_before = False
        self.dep_relations = []

        for line in open(path):
            line = line.strip()
            if line.startswith("#"):
                continue
            if len(line) == 0:
                if len(sentence.words) > 0:
                    self.sentences.append(sentence)
                    sentence = Sentence()
                continue
            items = line.split("\t")
            if not items[0].isdigit():
                continue
            word = ("" if space_before else " ") + items[1].strip()
            pos_tag = f"POS={items[3].strip()}" + ("" if items[5].strip() == "_" else f"|{items[5].strip()}")
            dependency = int(items[6].strip())
            dep_rel = items[7].strip()

            sentence.words.append(word)
            sentence.pos_tags.append(pos_tag)
            sentence.dependencies.append(dependency)
            sentence.dep_relations.append(dep_rel)
            self.dep_relations.append(dep_rel)
            space_before = "SpaceAfter=No" not in items[-1]
        
        if len(sentence.words) > 0:
            self.sentences.append(sentence)

        for sentence in self.sentences:
            encoding = tokenizer(sentence.words, add_special_tokens=True, is_split_into_words=True)
            sentence.subword_ids = torch.LongTensor(encoding.input_ids)
            sentence.subwords = tokenizer.convert_ids_to_tokens(encoding.input_ids)
            sentence.subword_to_word_map = encoding.word_ids()
            sentence.word_to_subword_map = torch.LongTensor([
                subword_index 
                for subword_index, word_index in enumerate(sentence.subword_to_word_map)
                if word_index is not None and word_index != sentence.subword_to_word_map[subword_index - 1]
            ])
            sentence.subword_to_word_map = torch.LongTensor([
                word_index if word_index is not None else -1 for word_index in sentence.subword_to_word_map
            ])
        
        if pos_ids_to_str is None:
            self.pos_ids_to_str = [
                pos_tag 
                for pos_tag, count in Counter(tag for sentence in self.sentences for tag in sentence.pos_tags).most_common()
            ]
        else:
            self.pos_ids_to_str = pos_ids_to_str

        self.pos_str_to_ids = {tag: i for i, tag in enumerate(self.pos_ids_to_str)}
        for sentence in self.sentences:
            sentence.pos_tag_ids = torch.LongTensor([self.pos_str_to_ids.get(tag, 0) for tag in sentence.pos_tags])

        if dep_ids_to_str is None:
            self.dep_ids_to_str = [dep_rel for dep_rel, count in Counter(self.dep_relations).most_common()]
        else:
            self.dep_ids_to_str = dep_ids_to_str
        self.dep_str_to_ids = {dep_rel: i for i, dep_rel in enumerate(self.dep_ids_to_str)}
        for sentence in self.sentences:
            sentence.dep_relation_ids = torch.LongTensor([self.dep_str_to_ids.get(dep_rel, 0) for dep_rel in sentence.dep_relations])

    def state_dict(self):
        return {
            "pos_vocabulary": self.pos_ids_to_str,
            "dep_vocabulary": self.dep_ids_to_str,
        }

    # load state dict
    def load_state_dict(self, state_dict):
        self.pos_ids_to_str = state_dict["pos_vocabulary"]
        self.pos_str_to_ids = {tag: i for i, tag in enumerate(self.pos_ids_to_str)}

        for sentence in self.sentences:
            sentence.pos_tag_ids = torch.LongTensor([self.pos_str_to_ids[tag] for tag in sentence.pos_tags])
            sentence.dep_relation_ids = torch.LongTensor([self.dep_str_to_ids[dep_rel] for dep_rel in sentence.dep_relations])

    def __getitem__(self, index: int):
        sentence = self.sentences[index]
        return {
            "words": sentence.words,
            "subword_ids": sentence.subword_ids,
            "pos_tag_ids": sentence.pos_tag_ids,
            "dependencies": torch.LongTensor(sentence.dependencies),
            "dep_relation_ids": torch.LongTensor(sentence.dep_relation_ids),
            "subword_to_word_map": sentence.subword_to_word_map,
            "word_to_subword_map": sentence.word_to_subword_map
        }

    def __len__(self):
        return len(self.sentences)