import sys
sys.path.append("/Users/naormatania/code/cyber-ai/NER/src/features")

from transformers import AutoTokenizer
from dnrti import DNRTIDataset
import time
import numpy as np
import json
import requests
from argparse import ArgumentParser
from nltk.tokenize import word_tokenize
from nltk.tokenize import TreebankWordTokenizer
import re

TOKENIZER_PATH = {
    'cyner': 'AI4Sec/cyner-xlm-roberta-base',
    'secner': 'CyberPeace-Institute/SecureBERT-NER',
}

PARENS_BRACKETS = (re.compile(r''), r' \g<0> ')

parser = ArgumentParser()
parser.add_argument('model', choices=['cyner', 'secner'])
parser.add_argument('--num_sentences', type=int, default=10)
args = parser.parse_args()

ds = DNRTIDataset(num_sentences=args.num_sentences)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH[args.model])

with open(f'test/results/iob_{args.model}.txt', 'w') as f:
    for i in range(len(ds)):
        text = ds[i]
        #r = requests.post(f'http://127.0.0.1:8000/ner/{args.model}/', data=json.dumps({'text': text}))
        #entities = r.json()['entities']
        #print(entities)
        # tokens = word_tokenize(text)
        tokenizer = TreebankWordTokenizer()
        tokenizer.PARENS_BRACKETS = PARENS_BRACKETS
        tokens = tokenizer.tokenize(text)
        for token in tokens:
            f.write(f'{token} O\n')
        f.write('\n')
