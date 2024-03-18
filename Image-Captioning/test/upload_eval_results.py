import os
import csv
from datasets import load_dataset
from PIL import Image
import pandas as pd
import shutil
from huggingface_hub import login
from argparse import ArgumentParser

MODELS = ['git-large', 'blip-large', 'blip-large-20', 'lblip-base', 'lblip-base-20', 'lblip-large-20']
REPORT_NAME_TO_DIR_PATH = {
  'desktop-ui': 'desktop-ui-dataset/images',
  'webscreen-test': 'website-screenshots/test',
}
CSV_FIELDS = ['file_name']  + [f'{model}_caption' for model in MODELS]

parser = ArgumentParser()
parser.add_argument('hf_key', type=str)
args = parser.parse_args()

login(args.hf_key)

for report_name, images_dir in REPORT_NAME_TO_DIR_PATH.items():
  combined_report = None
  for model in MODELS:
    df = pd.read_csv(f'reports/{report_name}-{model}.csv')
    df = df.rename(columns={'text': f'{model}_caption'})
    df = df.set_index('file_name')
    if combined_report is None:
       combined_report = df
    else:
       combined_report = pd.merge(combined_report, df, on="file_name")
  shutil.copytree(f'datasets/{images_dir}', 'tmpdir')
  combined_report.to_csv('tmpdir/metadata.csv')
  dataset = load_dataset('imagefolder', data_dir='tmpdir')
  dataset.push_to_hub(f'naorm/{report_name}-captions')
  shutil.rmtree('tmpdir')
