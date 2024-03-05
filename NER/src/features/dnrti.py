from ner_dataset import NERDataset
from datasets import load_dataset
import pandas as pd
import nltk
import os
import pathlib
import re
import shutil
from nltk.tokenize.treebank import TreebankWordDetokenizer
from huggingface_hub import login
from argparse import ArgumentParser

_LINE_RE = re.compile(r"((\S+)(\s+)?)+(O|(([IB])-(\S+)))$")

def prepare_dnrti_dataset():
    os.mkdir("DNRTI")

    for filepath in ['datasets/DNRTI/test.txt', 'datasets/DNRTI/train.txt', 'datasets/DNRTI/valid.txt']:
        lines = open(filepath, 'r').readlines()
        tokens = []
        for i, line in enumerate(lines):
            if line == "\n" or line == "O\n" or line == " O\n":
                continue
            tokens.append(re.match(_LINE_RE, line.rstrip().lstrip()).group(2))
        text = TreebankWordDetokenizer().detokenize(tokens)

        open(f"DNRTI/{os.path.basename(filepath)}", "w").write(text)

    dataset = load_dataset("text", data_dir="DNRTI")
    dataset.push_to_hub("naorm/DNRTI")
    shutil.rmtree('DNRTI')

class DNRTIDataset(NERDataset):
    def __init__(self, max_items = None, num_sentences = 1):
        ds_dnrti = load_dataset("naorm/DNRTI")
        train = ds_dnrti.data['train'].to_pandas()
        validation = ds_dnrti.data['validation'].to_pandas()
        test = ds_dnrti.data['test'].to_pandas()
        dnrti = pd.concat([train, validation, test])

        super().__init__(dnrti, max_items, num_sentences)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('hf_key', type=str)
    args = parser.parse_args()

    login(args.hf_key)
    prepare_dnrti_dataset()