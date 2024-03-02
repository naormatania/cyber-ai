import os
from datasets import load_dataset

DS_TYPES = ['train', 'validation', 'test']

os.system("git clone https://github.com/waltteri/desktop-ui-dataset.git")

ds = load_dataset("naorm/website-screenshots")

def download_dataset(ds, ds_type):
  worklist = ds[ds_type]

  os.makedirs(f'website-screenshots/{ds_type}')
  
  for i in range(len(worklist)):
    filename = f'file_{i}.jpg'
    worklist[i]['image'].save(f'website-screenshots/{ds_type}/{filename}')

for ds_type in DS_TYPES:
  download_dataset(ds, ds_type)
