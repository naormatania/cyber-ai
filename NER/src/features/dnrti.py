from ner_dataset import NERDataset
from datasets import load_dataset
import pandas as pd
import math
import os
import re
import shutil
from nltk.tokenize.treebank import TreebankWordDetokenizer
from huggingface_hub import login
from argparse import ArgumentParser
from torch.utils.data import Dataset

_LINE_RE = re.compile(r"^(((\S+)(\s+)?)+) (O|(([IB])-(\S+)))$")

def read_iob_tokens(full_path):
    lines = open(full_path, 'r').readlines()
    lines = [line for line in lines if line != "O\n" and line != " O\n"]
    tokens = []
    for line in lines:
        if line == "\n":
            if len(tokens) != 0:
                yield tokens
            tokens = []
            continue
        tokens.append(re.match(_LINE_RE, line).group(1))
    if len(tokens) != 0:
        yield tokens

def prepare_dnrti_dataset():
    os.mkdir("DNRTI")

    total_lines = []
    for name in ['test', 'train', 'valid']:
        full_path = f'datasets/DNRTI/{name}.txt'
        lines = open(full_path, 'r').readlines()
        lines = [line for line in lines if line != "O\n" and line != " O\n"]
        lines = [line.rstrip().lstrip() for line in lines]
        new_line = False
        for line in lines:
            if line == "":
                if not new_line:
                    total_lines.append("\n")
                    new_line = True
                continue
            new_line = False
            match_ = re.match(_LINE_RE, line)
            tokens = match_.group(1).split()
            class_ = match_.group(5)
            for token in tokens:
                total_lines.append(f"{token} {class_}\n")
    open("DNRTI/iob.txt", "w").writelines(total_lines)

    detokenizer = TreebankWordDetokenizer()
    sentences = '\n'.join([detokenizer.detokenize(tokens) for tokens in read_iob_tokens("DNRTI/iob.txt")])
    open("DNRTI/data.txt", "w").writelines(sentences)

class DNRTIDataset:
    def __init__(self, path = 'DNRTI', max_items = None, num_sentences = 1):
        self.num_sentences = num_sentences
        self.sentences = open(f'{path}/data.txt', 'r').readlines()
        if max_items is not None:
            self.sentences = self.sentences[:max_items]

    def __len__(self):
        return math.ceil(len(self.sentences)/self.num_sentences)

    def __getitem__(self, idx):
        sents = self.sentences[idx*self.num_sentences:idx*self.num_sentences+self.num_sentences]
        return ' '.join(sents)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('hf_key', type=str)
    args = parser.parse_args()

    login(args.hf_key)
    prepare_dnrti_dataset()