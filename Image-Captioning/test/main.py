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
import tritonclient.http as httpclient
from PIL import Image

FIELDS = ['file_name', 'text']

parser = ArgumentParser()
parser.add_argument('model', choices=['blip', 'blip2', 'blip-lavis', 'git', 'pix2struct-base', 'pix2struct-large', 'pix2struct'])
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

triton_client = httpclient.InferenceServerClient(url='127.0.0.1:8000', concurrency=1)

model_path = "models/blip-large/"
if args.model == 'blip2':
  model_path = "models/blip2/"
elif args.model == 'git':
  #model_path = "models/git-large/"
  model_path = "models/git-large-coco/"
elif args.model == 'pix2struct-base' or args.model == 'pix2struct':
  model_path = "models/pix2struct-base/"
elif args.model == 'pix2struct-large':
  model_path = "models/pix2struct-large/"
processor = AutoProcessor.from_pretrained(model_path)

def caption_batch(batch):
  before = time.time()
  files = [('files', (os.path.basename(path), open(path, 'rb'), 'image/jpeg')) for path in batch]
  if args.min_new_tokens:
    param_name = 'min_length' if args.model == 'blip-lavis' else 'min_new_tokens'
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

def triton_caption_batch(batch):
  before = time.time()
  captions = []
  for path in batch:
    image = np.array(Image.open(path).convert('RGB'))

    inputs = [httpclient.InferInput("image", image.shape, 'UINT8')]
    inputs[0].set_data_from_numpy(image)
    outputs = [httpclient.InferRequestedOutput('caption')]

    res = triton_client.infer(args.model, inputs, request_id="1", model_version="1", outputs=outputs)
    captions.append(res.as_numpy('caption')[0].decode())
  elapsed_time = time.time()-before
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
      if args.model == 'pix2struct':
        captions = triton_caption_batch(batch)
      else:
        captions = caption_batch(batch)
      writer.writerows([{'file_name': os.path.basename(path), 'text': caption} for path, caption in zip(batch, captions)])
      csvfile.flush()
else:
  for i in tqdm(range(0, len(image_paths), args.batch_size)):
    batch = image_paths[i:i+args.batch_size]
    if args.model == 'pix2struct':
      captions = triton_caption_batch(batch)
    else:
      caption = caption_batch(batch)

# TODO: add way to remove major outliers in time
print(f"avg_caption_size={np.mean(caption_size_arr)}, avg_time={np.mean(time_arr)}, std_time={np.std(time_arr)}, median_time={np.median(time_arr)}, 95pct_time={np.percentile(time_arr, 95)}")