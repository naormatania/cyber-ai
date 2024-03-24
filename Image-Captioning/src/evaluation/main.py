from datasets import load_dataset, Dataset
from transformers import BlipProcessor, BlipForConditionalGeneration, Blip2Processor, Blip2ForConditionalGeneration, AutoProcessor, AutoModelForCausalLM
from lavis.models import load_model_and_preprocess
import torch
import evaluate
import os
from tqdm import tqdm
from PIL import Image
import pandas as pd
from cider_scorer import CiderScorer
from huggingface_hub import login
from argparse import ArgumentParser

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 12

_ = torch.set_grad_enabled(False)

def hf_git_base():
    model_id = "microsoft/git-base"
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model = model.to(DEVICE)
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

def hf_git_large():
    model_id = "microsoft/git-large"
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model = model.to(DEVICE)
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

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

# Only loads on A100 that ~90GB RAM (V100 has ~13GB RAM which is not enough), unfortunately A100 is not available so much
def hf_blip2_opt_2_7b():
    model_id = "Salesforce/blip2-opt-2.7b"
    model = Blip2ForConditionalGeneration.from_pretrained(model_id)
    model = model.to(DEVICE)
    processor = Blip2Processor.from_pretrained(model_id)
    return model, processor

def lavis_blip_base():
    model, processor, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=DEVICE)
    return model, processor["eval"]

def lavis_blip_large():
    model, processor, _ = load_model_and_preprocess(name="blip_caption", model_type="large_coco", is_eval=True, device=DEVICE)
    return model, processor["eval"]

# Only loads on A100 that ~90GB RAM (V100 has ~13GB RAM which is not enough), unfortunately A100 is not available so much
def lavis_blip2_opt_2_7b():
    model, processor, _ = load_model_and_preprocess(name="blip2_opt", model_type="caption_coco_opt2.7b", is_eval=True, device=DEVICE)
    return model, processor["eval"]

MODEL_INITIALIZERS = {
    'hf-git-base': hf_git_base,
    'hf-git-large': hf_git_large,
    'hf-blip-base': hf_blip_base,
    'hf-blip-large': hf_blip_large,
    'lavis-blip-base': lavis_blip_base,
    'lavis-blip-large': lavis_blip_large,
}

def caption_images(model, processor, image_paths, min_new_tokens=None):
    predictions = []
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE)):
        images = [Image.open(path) for path in image_paths[i:i+BATCH_SIZE]]
        if model_name.startswith('hf'):
            inputs = processor(images=images, return_tensors="pt").to(DEVICE)
            out = model.generate(**inputs, max_new_tokens=500, min_new_tokens=min_new_tokens)
            predictions.extend(processor.batch_decode(out, skip_special_tokens=True))
        else:
            inputs = [processor(image).unsqueeze(0).to(DEVICE) for image in images]
            inputs = torch.stack(inputs).squeeze(1).to(DEVICE)
            if min_new_tokens is not None:
                out = model.generate({"image": inputs}, num_beams=1, max_length=500, min_length=min_new_tokens)
            else:
                out = model.generate({"image": inputs}, num_beams=1, max_length=500)
            predictions.extend(out)
    return predictions

def evaluate_metrics(predictions, references):
    metrics = {}
    for metric_name in ['sacrebleu', 'rouge', 'meteor']:
        metric = evaluate.load(metric_name)
        results = metric.compute(predictions=predictions, references=references)
        if metric_name == 'sacrebleu':
            metrics[metric_name] = results['score']
        elif metric_name == 'rouge':
            for k, v in results.items():
                metrics[k] = v
        elif metric_name == 'meteor':
            metrics[metric_name] = results['meteor']
    cider_scorer = CiderScorer()
    for pred, ref in zip(predictions, references):
        cider_scorer += (pred, ref)
    score, _ = cider_scorer.compute_score()
    metrics['CIDEr'] = score
    return metrics

parser = ArgumentParser()
parser.add_argument('hf_key', type=str)
args = parser.parse_args()

login(args.hf_key)

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
    print(f'Running model: {model_name}')
    model, processor = initializer()

    predictions = caption_images(model, processor, image_paths)
    total_predictions[model_name] = predictions
    metrics = evaluate_metrics(predictions, references)
    for metric_name, score in metrics.items():
        eval_results.append((model_name, metric_name, score))

    # predictions = caption_images(model, processor, image_paths, 20)
    # total_predictions[f'{model_name}/20'] = predictions
    # metrics = evaluate_metrics(predictions, references)
    # for metric_name, score in metrics.items():
    #     eval_results.append((f'{model_name}/20', metric_name, score))

eval_df = pd.DataFrame(eval_results, columns=['model_name', 'metric_name', 'value'])
ds = Dataset.from_pandas(eval_df)
ds.push_to_hub(f'naorm/caption-eval')

prediction_df = pd.DataFrame(total_predictions)
prediction_df.to_csv('content/coco/val2017/metadata.csv')
dataset = load_dataset('imagefolder', data_dir='content/coco/val2017')
dataset.push_to_hub(f'naorm/all-captions')