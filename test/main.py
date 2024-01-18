from src.features.malware_text_db import MalwareTextDataset
import os
import time
import numpy as np

ds = MalwareTextDataset(max_items=1000, num_sentences=10)

time_arr = []

for i in range(len(ds)):
    text = ds[i]
    cmd = f'curl http://127.0.0.1:8000/ner/secner/ -H "Content-Type: application/json" -v -d \'{"text": "{text}"}\''
    before = time.time()
    os.system(cmd)
    time_arr.append(time.time()-before)

print(f"avg_time={np.mean(time_arr)}, std_time={np.std(time_arr)}")