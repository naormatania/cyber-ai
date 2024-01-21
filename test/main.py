import sys
sys.path.append("/Users/naormatania/code/cyber-ner/src/features")

from transformers import AutoTokenizer
from malware_text_db import MalwareTextDataset
import time
import os
import numpy as np
import json
import requests

ds = MalwareTextDataset(num_sentences=10)

time_arr = []
padding_precentage_arr = []
num_entities = 0

for i in range(len(ds)):
    text = ds[i]
    
    tokenizer = AutoTokenizer.from_pretrained('CyberPeace-Institute/SecureBERT-NER')
    encode = tokenizer.encode_plus(text)
    num_tokens = len(encode['input_ids'])
    if num_tokens <= 512:
        padding_precentage = (512 - len(encode['input_ids'])) / 512
    else:
        padding_precentage = (1024 - len(encode['input_ids'])) / 1024
    padding_precentage_arr.append(padding_precentage)

    before = time.time()
    r = requests.post('http://127.0.0.1:8000/ner/secner/', data=json.dumps({'text': text}))
    time_arr.append(time.time()-before)
    num_entities = num_entities + len(r.json()['entities'])

print(f"avg_time={np.mean(time_arr)}, std_time={np.std(time_arr)}, num_entities={num_entities}, padding_precentage={np.mean(padding_precentage_arr)}")