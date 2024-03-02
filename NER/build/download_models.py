from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig
import os

def download_model(model_path, model_name):
    # Check if the directory already exists
    if not os.path.exists(model_path):
        # Create the directory
        os.makedirs(model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)

    # Save the model, tokenizer and config to the specified directory
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    config.save_pretrained(model_path)

download_model('models/cyner/', 'AI4Sec/cyner-xlm-roberta-base')
download_model('models/SecureBERT-NER/', 'CyberPeace-Institute/SecureBERT-NER')