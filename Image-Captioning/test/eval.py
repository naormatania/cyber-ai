import os
from datasets import load_dataset
from argparse import ArgumentParser
import evaluate
import requests
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument('model', choices=['blip', 'blip-lavis', 'git'])
parser.add_argument('metric', choices=['sacrebleu', 'rouge'])
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--min_new_tokens', type=int, nargs='?')

args = parser.parse_args()

def caption_batch(batch):
  files = [('files', (os.path.basename(path), open(path, 'rb'), 'image/jpeg')) for path in batch]
  if args.min_new_tokens:
    param_name = 'min_length' if args.model == 'blip-lavis' else 'min_new_tokens'
    r = requests.post(f'http://127.0.0.1:8000/caption/{args.model}?{param_name}={args.min_new_tokens}', files=files)
  else:
    r = requests.post(f'http://127.0.0.1:8000/caption/{args.model}', files=files)
  captions = r.json()['captions'][0]
  return captions


ds = load_dataset("ydshieh/coco_dataset_script", "2017", data_dir=os.path.abspath("datasets/coco"))['validation']
total_predictions = []
total_references = []

TOTAL_SIZE = 360 # len(ds)
for i in tqdm(range(0, TOTAL_SIZE, args.batch_size)):
    batch = ds[i:i+args.batch_size]
    total_predictions.extend(caption_batch(batch['image_path']))
    total_references.extend([[caption] for caption in batch['caption']])

metric = evaluate.load(args.metric)
results = metric.compute(predictions=total_predictions, references=total_references)
if args.metric == 'sacrebleu':
  print(f"BLEU score: {results['score']}")
elif args.metric == 'rouge':
  print(f"ROUGE score: {results}")
