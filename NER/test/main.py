import sys
sys.path.append("/Users/naormatania/code/cyber-ai/NER/src/features")

from transformers import AutoTokenizer
from dnrti import DNRTIDataset
import time
import numpy as np
import json
import requests
from argparse import ArgumentParser

TOKENIZER_PATH = {
    'cyner': 'AI4Sec/cyner-xlm-roberta-base',
    'secner': 'CyberPeace-Institute/SecureBERT-NER',
}

parser = ArgumentParser()
parser.add_argument('model', choices=['cyner', 'secner'])
parser.add_argument('--num_sentences', type=int, default=10)
args = parser.parse_args()

ds = DNRTIDataset(max_items=10, num_sentences=args.num_sentences)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH[args.model])

for i in range(len(ds)):
    text = ds[i]
    
    r = requests.post(f'http://127.0.0.1:8000/ner/{args.model}/', data=json.dumps({'text': text}))
    entities = r.json()['entities']
    print(entities)
