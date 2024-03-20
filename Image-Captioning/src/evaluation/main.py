from datasets import load_dataset
from transformers import BlipProcessor, BlipForConditionalGeneration, Blip2Processor, Blip2ForConditionalGeneration
import torch
import evaluate
import os
from tqdm import tqdm
from PIL import Image
from cider_scorer import CiderScorer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 12

def hf_blip_base():
    model_id = "Salesforce/blip-image-captioning-base"
    model = BlipForConditionalGeneration.from_pretrained(model_id)
    model = model.to(DEVICE)
    processor = BlipProcessor.from_pretrained(model_id)
    return model, processor

def hf_blip_large():
    model_id = "Salesforce/blip-image-captioning-large"
    model = BlipForConditionalGeneration.from_pretrained(model_id)
    model = model.to(DEVICE)
    processor = BlipProcessor.from_pretrained(model_id)
    return model, processor

def hf_blip2_opt_2_7b():
    model_id = "Salesforce/blip2-opt-2.7b"
    model = Blip2ForConditionalGeneration.from_pretrained(model_id)
    model = model.to(DEVICE)
    processor = Blip2Processor.from_pretrained(model_id)
    return model, processor

MODEL_INITIALIZERS = {
   'hf-blip-base': hf_blip_base,
#    'hf-blip-large': hf_blip_large,
#    'hf-blip2-opt-2.7b': hf_blip2_opt_2_7b,
}

def caption_images(model, processor, image_paths):
    predictions = []
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE)):
        images = [Image.open(path) for path in image_paths[i:i+BATCH_SIZE]]
        inputs = processor(images=images, return_tensors="pt").to(DEVICE)
        out = model.generate(**inputs, max_new_tokens=500)
        predictions.extend(processor.batch_decode(out, skip_special_tokens=True))
    return predictions

ds = load_dataset("ydshieh/coco_dataset_script", "2017", data_dir="/content/coco")

image_path_to_captions = {}
for m in ds['validation']:
  image_path = m['image_path']
  caption = m['caption']
  if image_path in image_path_to_captions:
    image_path_to_captions[image_path].append(caption)
  else:
    image_path_to_captions[image_path] = [caption]
image_path_to_captions = list(image_path_to_captions.items())
references = [item[1][:5] for item in image_path_to_captions]
image_paths = [item[0] for item in image_path_to_captions]

total_predictions = {'file_name': list(map(os.path.basename, image_paths))}
eval_results = []

for model_name, initializer in MODEL_INITIALIZERS.items():
    model, processor = initializer()
    predictions = caption_images(model, processor, image_paths)
    total_predictions[model_name] = predictions
    for metric_name in ['sacrebleu', 'rouge', 'meteor']:
        metric = evaluate.load(metric_name)
        results = metric.compute(predictions=predictions, references=references)
        if metric_name == 'sacrebleu':
            eval_results.append((model_name, metric_name, results['score']))
        elif metric_name == 'rouge':
            for k, v in results.items():
                eval_results.append((model_name, k, v))
        elif metric_name == 'meteor':
            eval_results.append((model_name, metric_name, results['meteor']))
    cider_scorer = CiderScorer()
    for pred, ref in zip(predictions, references):
        cider_scorer += (pred, ref)
    score, _ = cider_scorer.compute_score()
    eval_results.append((model_name, 'CIDEr', score))

print(eval_results)