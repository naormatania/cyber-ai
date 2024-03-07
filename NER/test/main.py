import sys
sys.path.append("/Users/naormatania/code/cyber-ai/NER/src/features")

from transformers import AutoTokenizer
from dnrti import DNRTIDataset, read_iob_tokens
import time
import numpy as np
import json
import requests
from argparse import ArgumentParser
from nltk.tokenize import word_tokenize, TreebankWordTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
import re
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument('model', choices=['cyner', 'secner'])
parser.add_argument('--num_sentences', type=int, default=10)
args = parser.parse_args()

ds = DNRTIDataset(num_sentences=args.num_sentences)

list_tokens = read_iob_tokens("DNRTI/iob.txt")
detokenizer = TreebankWordDetokenizer()

with open(f'test/results/iob_{args.model}.txt', 'w') as f:
    for tokens in tqdm(list(list_tokens)):
        text = detokenizer.detokenize(tokens)
        text_parts = text.split()
        r = requests.post(f'http://127.0.0.1:8000/ner/{args.model}/', data=json.dumps({'text': text}))
        entities = r.json()['entities']
        
        token_index = 0
        entity_index = 0
        for i, text_part in enumerate(text_parts):
            j = 0
            while detokenizer.detokenize(tokens[token_index:token_index+1+j]) != text_part:
                j = j + 1
            token_type = 'O'
            while entity_index < len(entities) and i > entities[entity_index][3]:
                entity_index = entity_index + 1
            if entity_index < len(entities):
                if i >= entities[entity_index][2] and i <= entities[entity_index][3]:
                    token_type = entities[entity_index][0]
            for t, token in enumerate(tokens[token_index:token_index+1+j]):
                if token_type != 'O':
                    if t == 0:
                        f.write(f'{token} B-{token_type}\n')
                    else:
                        f.write(f'{token} I-{token_type}\n')
                else:
                    f.write(f'{token} O\n')
            token_index = token_index + j + 1
        f.write('\n')
        
