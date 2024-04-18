# AI for Cyber purposes

This repository includes the following:
1. NER - extract cyber entities. Evaluation of different models
2. Image-Captioning - caption desktop, screenshots and regular images. Evaluation of different models
3. Playground - Export the selected NER and Image-Captioning models via streamlit
4. Triton - Incomplete efforts to improve models inference times using Nvidia Triton

## Run the playground
1. In `NER` folder:
```
./build/export.sh
```
2. In `Image-Captioning` folder:
```
./build/download_lavis_model.sh
```
3. In `Image-Captioning/build/` folder:
```
git clone git@github.com:huggingface/optimum.git
```
4. In `Image-Captioning/build/optimum` folder:
```
python3 setup.py develop
cp /usr/local/bin/optimum-cli .
```
5. In `Image-Captioning/build` folder:
```
./optimum/optimum-cli export onnx -m google/pix2struct-screen2words-base --optimize O3 onnx/pix2struct_base
```
6. In root folder:
```
docker-compose build && docker-compose up
```
7. Access the playground via http://0.0.0.0:8501/