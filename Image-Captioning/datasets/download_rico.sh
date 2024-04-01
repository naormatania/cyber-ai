wget https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/unique_uis.tar.gz
tar --wildcards -xvzf unique_uis.tar.gz '*.jpg'
mv combined images
git clone https://github.com/google-research-datasets/screen2words.git