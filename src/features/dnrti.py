from ner_dataset import NERDataset
from datasets import load_dataset
import pandas as pd
import nltk
import os
import pathlib
import re
from nltk.tokenize.treebank import TreebankWordDetokenizer

_LINE_RE = re.compile(r"((\S+)(\s+)?)+([IOB])-?(\S+)?")

def prepare_dnrti_dataset():
    os.system("curl -L -O https://github.com/SCreaMxp/DNRTI-A-Large-scale-Dataset-for-Named-Entity-Recognition-in-Threat-Intelligence/raw/master/DNRTI.rar")
    os.system("unrar x DNRTI.rar")
    os.mkdir("DNRTI")

    for filepath in ['test.txt', 'train.txt', 'valid.txt']:
        lines = open(filepath, 'r').readlines()
        tokens = []
        for i, line in enumerate(lines):
            if line == "\n" or line == "O\n" or line == " O\n":
                continue
            tokens.append(re.match(_LINE_RE, line).group(1).rstrip())
        text = TreebankWordDetokenizer().detokenize(tokens)

        open(f"DNRTI/{os.path.basename(filepath)}", "w").write(text)

    dataset = load_dataset("text", data_dir="DNRTI")
    dataset.push_to_hub("naorm/DNRTI")

class DNRTIDataset(NERDataset):
    def __init__(self, max_items = None, num_sentences = 1):
        ds_dnrti = load_dataset("naorm/DNRTI")
        train = ds_dnrti.data['train'].to_pandas()
        validation = ds_dnrti.data['validation'].to_pandas()
        test = ds_dnrti.data['test'].to_pandas()
        dnrti = pd.concat([train, validation, test])

        super().__init__(dnrti, max_items, num_sentences)