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

_LINE_RE = re.compile(r"((\S+)(\s+)?)+(O|(([IB])-(\S+)))$")

def prepare_dnrti_dataset():
    os.mkdir("DNRTI")

    total_lines = []
    total_sentences = []
    for name in ['test', 'train', 'valid']:
        full_path = f'datasets/DNRTI/{name}.txt'
        lines = open(full_path, 'r').readlines()
        lines = [line for line in lines if line != "O\n" and line != " O\n"]
        sentences = []
        tokens = []
        for line in lines:
            if line == "\n":
                sentences.append(TreebankWordDetokenizer().detokenize(tokens) + "\n")
                tokens = []
                continue
            tokens.append(re.match(_LINE_RE, line.rstrip().lstrip()).group(2))
        sentences.append(TreebankWordDetokenizer().detokenize(tokens))
        total_lines.extend(lines)
        total_sentences.extend(sentences)
    open(f"DNRTI/iob.txt", "w").writelines(lines)
    open(f"DNRTI/data.txt", "w").writelines(sentences)

    # dataset = load_dataset("text", data_dir="DNRTI")
    # dataset.push_to_hub("naorm/DNRTI")
    # shutil.rmtree('DNRTI')

class DNRTIDataset(Dataset):
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