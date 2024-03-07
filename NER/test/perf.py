import sys
sys.path.append("/Users/naormatania/code/cyber-ai/NER/src/features")

from transformers import AutoTokenizer
from malware_text_db import MalwareTextDataset
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

ds = MalwareTextDataset(num_sentences=args.num_sentences)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH[args.model])

time_arr = []
padding_precentage_arr = []
total_num_entities = 0

for i in range(len(ds)):
    text = ds[i]
    encode = tokenizer.encode_plus(text)
    num_tokens = len(encode['input_ids'])
    if num_tokens <= 512:
        padding_precentage = (512 - len(encode['input_ids'])) / 512
    else:
        padding_precentage = (1024 - len(encode['input_ids'])) / 1024
    padding_precentage_arr.append(padding_precentage)

    before = time.time()
    r = requests.post(f'http://127.0.0.1:8000/ner/{args.model}/', data=json.dumps({'text': text}))
    elapsed_time = time.time()-before
    time_arr.append(elapsed_time)
    num_entities = len(r.json()['entities'])
    total_num_entities = total_num_entities + num_entities
    print(f"elapsed_time={elapsed_time}, num_entities={num_entities}, padding_precentage={padding_precentage:.0%}")

print(f"avg_time={np.mean(time_arr)}, std_time={np.std(time_arr)}, median_time={np.median(time_arr)}, 95pct_time={np.percentile(time_arr, 95)}, total_num_entities={total_num_entities}, padding_precentage={np.mean(padding_precentage_arr):.0%}")