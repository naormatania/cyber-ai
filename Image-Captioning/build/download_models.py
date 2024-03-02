from transformers import BlipProcessor, BlipForConditionalGeneration, Blip2Processor, Blip2ForConditionalGeneration
from transformers import AutoModelForCausalLM, AutoProcessor
from argparse import ArgumentParser
import os

def download_blip_model(model_path, model_name):
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)

    # Save the processor and model to the specified directory
    processor.save_pretrained(model_path)
    model.save_pretrained(model_path)

def download_git_model(model_path, model_name):
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Save the processor and model to the specified directory
    processor.save_pretrained(model_path)
    model.save_pretrained(model_path)

def download_blip2_model(model_path, model_name):
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(model_name)

    # Save the processor and model to the specified directory
    processor.save_pretrained(model_path)
    model.save_pretrained(model_path)

parser = ArgumentParser()
parser.add_argument('model', choices=['blip', 'git', 'blip2'])
args = parser.parse_args()

if args.model == 'blip':
    download_blip_model('models/blip-base/', 'Salesforce/blip-image-captioning-base')
    download_blip_model('models/blip-large/', 'Salesforce/blip-image-captioning-large')
elif args.model == 'git':
    download_git_model('models/git-base/', 'microsoft/git-base')
    download_git_model('models/git-large/', 'microsoft/git-large')
    download_git_model('models/git-large-textcaps/', 'microsoft/git-large-textcaps')
elif args.model == 'blip2':
    download_blip2_model('models/blip2/', 'Salesforce/blip2-opt-2.7b')
