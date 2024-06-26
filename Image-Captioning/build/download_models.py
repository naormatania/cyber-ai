from transformers import BlipProcessor, BlipForConditionalGeneration, Blip2Processor, Blip2ForConditionalGeneration, Pix2StructForConditionalGeneration, Pix2StructProcessor
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

def download_pix2struct_model(model_path, model_name):
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    processor = Pix2StructProcessor.from_pretrained(model_name, is_vqa=False)
    model = Pix2StructForConditionalGeneration.from_pretrained(model_name)

    # Save the processor and model to the specified directory
    processor.save_pretrained(model_path)
    model.save_pretrained(model_path)

parser = ArgumentParser()
parser.add_argument('model', choices=['blip', 'git', 'blip2', 'lavis-blip', 'pix2struct'])
args = parser.parse_args()

if args.model == 'blip':
    download_blip_model('models/blip-base/', 'Salesforce/blip-image-captioning-base')
    download_blip_model('models/blip-large/', 'Salesforce/blip-image-captioning-large')
elif args.model == 'git':
    # download_git_model('models/git-base/', 'microsoft/git-base')
    # download_git_model('models/git-large/', 'microsoft/git-large')
    # download_git_model('models/git-large-textcaps/', 'microsoft/git-large-textcaps')
    download_git_model('models/git-base-coco/', 'microsoft/git-base-coco')
    download_git_model('models/git-large-coco/', 'microsoft/git-large-coco')
elif args.model == 'blip2':
    download_blip2_model('models/blip2/', 'Salesforce/blip2-opt-2.7b')
elif args.model == 'lavis-blip':
    os.system('cd models; wget https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP/blip_coco_caption_base.pth')
    os.system('cd models; wget https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth')
elif args.model == 'pix2struct':
    download_pix2struct_model('models/pix2struct-base/', 'google/pix2struct-screen2words-base')
    # download_pix2struct_model('models/pix2struct-large/', 'google/pix2struct-screen2words-large')