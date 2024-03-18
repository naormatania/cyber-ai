from datasets import load_dataset
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import evaluate
from tqdm import tqdm
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 12
BLIP_MODEL_ID = "Salesforce/blip-image-captioning-large"

BLIP_MODEL = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_ID)
BLIP_MODEL = BLIP_MODEL.to(DEVICE)
BLIP_PROCESSOR = BlipProcessor.from_pretrained(BLIP_MODEL_ID)    

ds = load_dataset("ydshieh/coco_dataset_script", "2017", data_dir="/content/coco")

image_path_to_captions = {}
for m in ds['validation']:
  image_path = m['image_path']
  caption = m['caption']
  if image_path in image_path_to_captions:
    image_path_to_captions[image_path].append(caption)
  else:
    image_path_to_captions[image_path] = [caption]

total_predictions = []
total_references = []

image_path_to_captions = image_path_to_captions.items()
for i in tqdm(range(0, len(image_path_to_captions), BATCH_SIZE)):
    batch = image_path_to_captions[i:i+BATCH_SIZE]
    images = [Image.open(item[0]) for item in batch]
    inputs = BLIP_PROCESSOR(images=batch, return_tensors="pt").to(DEVICE)
    out = BLIP_MODEL.generate(**inputs, max_new_tokens=500)
    total_predictions.extend(BLIP_PROCESSOR.batch_decode(out, skip_special_tokens=True))
    total_references.extend([item[1] for item in batch])

for metric_name in ['sacrebleu', 'rouge']:
  metric = evaluate.load(metric_name)
  results = metric.compute(predictions=total_predictions, references=total_references)
  if metric_name == 'sacrebleu':
    print(f"BLEU score: {results['score']}")
  elif metric_name == 'rouge':
    print(f"ROUGE score: {results}")