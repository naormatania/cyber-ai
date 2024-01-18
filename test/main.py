import sys
sys.path.append("/Users/naormatania/code/cyber-ner/src/features")

from malware_text_db import MalwareTextDataset
import time
import os
import numpy as np
import json
import requests

ds = MalwareTextDataset(num_sentences=10)

time_arr = []
num_entities = 0

for i in range(len(ds)):
    text = ds[i]
    before = time.time()
    r = requests.post('http://127.0.0.1:8000/ner/secner/', data=json.dumps({'text': text}))
    time_arr.append(time.time()-before)
    num_entities = num_entities + len(r.json()['entities'])

print(f"avg_time={np.mean(time_arr)}, std_time={np.std(time_arr)}, num_entities={num_entities}")