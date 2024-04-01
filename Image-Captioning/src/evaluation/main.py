from datasets import load_dataset, Dataset
from transformers import BlipProcessor, BlipForConditionalGeneration, Blip2Processor, Blip2ForConditionalGeneration, AutoProcessor, AutoModelForCausalLM, Pix2StructForConditionalGeneration, Pix2StructProcessor
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
import gc
import csv

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 12
BLIP2_BATCH_SIZE = 1

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

def hf_git_base_coco():
    model_id = "microsoft/git-base-coco"
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model = model.to(DEVICE)
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

def hf_git_large_coco():
    model_id = "microsoft/git-large-coco"
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

def hf_blip2_opt_2_7b():
    model_id = "Salesforce/blip2-opt-2.7b"
    model = Blip2ForConditionalGeneration.from_pretrained(model_id, device_map="auto")
    processor = Blip2Processor.from_pretrained(model_id)
    return model, processor

def hf_blip2_opt_2_7b_16bit():
    model_id = "Salesforce/blip2-opt-2.7b"
    model = Blip2ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    processor = Blip2Processor.from_pretrained(model_id)
    return model, processor

def hf_blip2_opt_2_7b_8bit():
    model_id = "Salesforce/blip2-opt-2.7b"
    model = Blip2ForConditionalGeneration.from_pretrained(model_id, load_in_8bit=True, device_map="auto")
    processor = Blip2Processor.from_pretrained(model_id)
    return model, processor

def hf_blip2_opt_2_7b_coco():
    model_id = "Salesforce/blip2-opt-2.7b-coco"
    model = Blip2ForConditionalGeneration.from_pretrained(model_id, device_map="auto")
    processor = Blip2Processor.from_pretrained(model_id)
    return model, processor

def hf_blip2_opt_2_7b_coco_16bit():
    model_id = "Salesforce/blip2-opt-2.7b-coco"
    model = Blip2ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    processor = Blip2Processor.from_pretrained(model_id)
    return model, processor

def hf_blip2_opt_2_7b_coco_8bit():
    model_id = "Salesforce/blip2-opt-2.7b-coco"
    model = Blip2ForConditionalGeneration.from_pretrained(model_id, load_in_8bit=True, device_map="auto")
    processor = Blip2Processor.from_pretrained(model_id)
    return model, processor

def hf_pix2struct_screen2words_base():
    model_id = "google/pix2struct-screen2words-base"
    model = Pix2StructForConditionalGeneration.from_pretrained(model_id)
    model = model.to(DEVICE)
    processor = Pix2StructProcessor.from_pretrained(model_id)
    return model, processor

def hf_pix2struct_screen2words_large():
    model_id = "google/pix2struct-screen2words-large"
    model = Pix2StructForConditionalGeneration.from_pretrained(model_id)
    model = model.to(DEVICE)
    processor = Pix2StructProcessor.from_pretrained(model_id)
    return model, processor

BASE_MODEL_INITIALIZERS = {
    'hf-git-base': hf_git_base,
    'hf-git-large': hf_git_large,
    'hf-git-base-coco': hf_git_base_coco,
    'hf-git-large-coco': hf_git_large_coco,
    'hf-blip-base': hf_blip_base,
    'hf-blip-large': hf_blip_large,
    'lavis-blip-base': lavis_blip_base,
    'lavis-blip-large': lavis_blip_large,
}

BLIP2_MODEL_INITIALIZERS = {
    'hf-blip2': hf_blip2_opt_2_7b,
    'hf-blip2-coco': hf_blip2_opt_2_7b_coco,
}

QUANT_BLIP2_MODEL_INITIALIZERS = {
    'hf-blip2-16bit': hf_blip2_opt_2_7b_16bit,
    'hf-blip2-8bit': hf_blip2_opt_2_7b_8bit,
    'hf-blip2-coco-16bit': hf_blip2_opt_2_7b_coco_16bit,
    'hf-blip2-coco-8bit': hf_blip2_opt_2_7b_coco_8bit,
}

PIX2STRUCT_MODEL_INITIALIZERS = {
    'hf-pix2struct-base': hf_pix2struct_screen2words_base,
    'hf-pix2struct-large': hf_pix2struct_screen2words_large,
}

def caption_images(model_name, model, processor, image_paths, min_new_tokens=None):
    predictions = []
    batch_size = BLIP2_BATCH_SIZE if 'blip2' in model_name and 'bit' not in model_name else BATCH_SIZE
    for i in tqdm(range(0, len(image_paths), batch_size)):
        images = [Image.open(path) for path in image_paths[i:i+batch_size]]
        if model_name.startswith('hf'):
            if model_name.endswith('bit'):
                inputs = processor(images=images, return_tensors="pt").to(DEVICE, torch.float16)
            else:
                inputs = processor(images=images, return_tensors="pt").to(DEVICE)
            out = model.generate(**inputs, max_new_tokens=500, min_new_tokens=min_new_tokens)
            predictions.extend([s.strip() for s in processor.batch_decode(out, skip_special_tokens=True)])
        else:
            inputs = [processor(image).unsqueeze(0).to(DEVICE) for image in images]
            inputs = torch.stack(inputs).squeeze(1).to(DEVICE)
            if min_new_tokens is not None:
                out = model.generate({"image": inputs}, num_beams=1, max_length=500, min_length=min_new_tokens)
            else:
                out = model.generate({"image": inputs}, num_beams=1, max_length=500)
            predictions.extend([s.strip() for s in out])
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

def load_coco():
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
    return references, image_paths

def load_screen2words():
    csvfile = open('/content/screen2words/screen_summaries.csv')
    reader = csv.reader(csvfile, delimiter=',')
    next(reader, None)  # skip the headers
    screen_id_to_captions = {}
    for (screen_id, caption) in reader:
        if screen_id in screen_id_to_captions:
            screen_id_to_captions[screen_id].append(caption)
        else:
            screen_id_to_captions[screen_id] = [caption]
    test_screen_ids = [l.strip() for l in open('/content/screen2words/split/test_screens.txt','r').readlines()]
    screen_id_to_captions = {screen_id: captions for screen_id, captions in screen_id_to_captions.items() if screen_id in test_screen_ids}
    references = []
    image_paths = []
    for screen_id, captions in screen_id_to_captions.items():
        references.append(captions)
        image_paths.append(f'images/{screen_id}.jpg')
    for file_name in os.listdir('/content/images'): 
        if os.path.splitext(file_name)[0] not in screen_id_to_captions:
            os.remove(f'/content/images/{file_name}')
    return references, image_paths

def evaluate_models(model_initializers, references, image_paths):
    total_predictions = {'file_name': list(map(os.path.basename, image_paths))}
    eval_results = []

    for model_name, initializer in model_initializers.items():
        print(f'Running model: {model_name}')
        model, processor = initializer()

        predictions = caption_images(model_name, model, processor, image_paths)
        total_predictions[model_name] = predictions
        metrics = evaluate_metrics(predictions, references)
        for metric_name, score in metrics.items():
            eval_results.append((model_name, metric_name, score))
        
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()

        # predictions = caption_images(model, processor, image_paths, 20)
        # total_predictions[f'{model_name}/20'] = predictions
        # metrics = evaluate_metrics(predictions, references)
        # for metric_name, score in metrics.items():
        #     eval_results.append((f'{model_name}/20', metric_name, score))
    
    return total_predictions, eval_results

def save_eval_results(eval_results, dataset_suffix = ''):
    eval_df = pd.DataFrame(eval_results, columns=['model_name', 'metric_name', 'value'])
    ds = Dataset.from_pandas(eval_df)
    ds.push_to_hub(f'naorm/caption-eval{dataset_suffix}')

def save_predictions(predictions, data_dir, dataset_suffix = ''):
    prediction_df = pd.DataFrame(predictions)
    prediction_df.to_csv(f'{data_dir}/metadata.csv', index=False)
    dataset = load_dataset('imagefolder', data_dir=data_dir)
    dataset.push_to_hub(f'naorm/all-captions{dataset_suffix}')

parser = ArgumentParser()
parser.add_argument('hf_key', type=str)
parser.add_argument('dataset', choices=['coco', 'screen2words'])
args = parser.parse_args()

login(args.hf_key)

if args.dataset == 'coco':
    references, image_paths = load_coco()

    print("Evaluate base models:")
    predictions, eval_results = evaluate_models(BASE_MODEL_INITIALIZERS, references, image_paths)
    save_eval_results(eval_results, '-coco')
    save_predictions(predictions, '/content/coco/val2017', '-coco')

    print("Evaluate blip2 models:")
    predictions, eval_results = evaluate_models(BLIP2_MODEL_INITIALIZERS, references, image_paths)
    save_eval_results(eval_results, '-coco-blip2')
    save_predictions(predictions, '/content/coco/val2017', '-coco-blip2')

    print("Evaluate blip2-quant models:")
    predictions, eval_results = evaluate_models(QUANT_BLIP2_MODEL_INITIALIZERS, references, image_paths)
    save_eval_results(eval_results, '-coco-blip2-quant')
    save_predictions(predictions, '/content/coco/val2017', '-coco-blip2-quant')
else:
    references, image_paths = load_screen2words()

    print("Evaluate base models:")
    predictions, eval_results = evaluate_models(BASE_MODEL_INITIALIZERS, references, image_paths)
    save_eval_results(eval_results, '-screen2words')
    save_predictions(predictions, '/content/images', '-screen2words')

    print("Evaluate blip2 models:")
    predictions, eval_results = evaluate_models(BLIP2_MODEL_INITIALIZERS, references, image_paths)
    save_eval_results(eval_results, '-screen2words-blip2')
    save_predictions(predictions, '/content/images', '-screen2words-blip2')

    print("Evaluate blip2-quant models:")
    predictions, eval_results = evaluate_models(QUANT_BLIP2_MODEL_INITIALIZERS, references, image_paths)
    save_eval_results(eval_results, '-screen2words-blip2-quant')
    save_predictions(predictions, '/content/images', '-screen2words-blip2-quant')

    print("Evaluate pix2struct models:")
    predictions, eval_results = evaluate_models(PIX2STRUCT_MODEL_INITIALIZERS, references, image_paths)
    save_eval_results(eval_results, '-screen2words-pix2struct')
    save_predictions(predictions, '/content/images', '-screen2words-pix2struct')
