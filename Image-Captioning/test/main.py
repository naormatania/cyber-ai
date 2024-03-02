import os
import time
import numpy as np
import requests
from argparse import ArgumentParser
import os
import csv
import json
from tqdm import tqdm
from transformers import AutoProcessor

FIELDS = ['file_name', 'text']

parser = ArgumentParser()
parser.add_argument('model', choices=['blip', 'blip2', 'blip2-lavis', 'git'])
parser.add_argument('dataset', choices=['desktop-ui-dataset/images', 'website-screenshots/train', 'website-screenshots/validation', 'website-screenshots/test'])
parser.add_argument('--report_name', type=str, default="")
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--min_new_tokens', type=int, nargs='?')
args = parser.parse_args()

images_dir = f'datasets/{args.dataset}'
images = os.listdir(images_dir)
images.sort()
image_paths = [os.path.join(images_dir, image) for image in images]

time_arr = []
caption_size_arr = []

model_path = "models/blip-large/"
if args.model == 'blip2':
  model_path = "models/blip2/"
elif args.model == 'git':
  model_path = "models/git-large/"
processor = AutoProcessor.from_pretrained(model_path)

def caption_batch(batch):
  before = time.time()
  files = [('files', (os.path.basename(path), open(path, 'rb'), 'image/jpeg')) for path in batch]
  if args.min_new_tokens:
    param_name = 'min_length' if args.model == 'blip2-lavis' else 'min_new_tokens'
    r = requests.post(f'http://127.0.0.1:8000/caption/{args.model}?{param_name}={args.min_new_tokens}', files=files)
  else:
    r = requests.post(f'http://127.0.0.1:8000/caption/{args.model}', files=files)    
  elapsed_time = time.time()-before
  captions = r.json()['captions'][0]
  print(captions)
  print(f"elapsed_time={elapsed_time}, batch_size={len(batch)}")
  time_arr.append(elapsed_time/len(batch))
  caption_size_arr.extend([len(processor.tokenizer.tokenize(caption)) for caption in captions])
  return captions

if args.report_name:
  with open(f'reports/{args.report_name}.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=FIELDS)
    writer.writeheader()
    for i in tqdm(range(0, len(image_paths), args.batch_size)):
      batch = image_paths[i:i+args.batch_size]
      captions = caption_batch(batch)
      writer.writerows([{'file_name': os.path.basename(path), 'text': caption} for path, caption in zip(batch, captions)])
      csvfile.flush()
else:
  for i in tqdm(range(0, len(image_paths), args.batch_size)):
    batch = image_paths[i:i+args.batch_size]
    caption = caption_batch(batch)

print(f"avg_caption_size={np.mean(caption_size_arr)}, avg_time={np.mean(time_arr)}, std_time={np.std(time_arr)}, median_time={np.median(time_arr)}, 95pct_time={np.percentile(time_arr, 95)}")