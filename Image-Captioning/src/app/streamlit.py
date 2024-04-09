from lavis.models import load_model_and_preprocess, load_preprocess
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
from PIL import Image
import pandas as pd
import streamlit as st
from lavis.common.registry import registry
from omegaconf import OmegaConf
import torch
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(layout="wide")

@st.cache
def init_pytorch():
    _ = torch.set_grad_enabled(False)
    torch.set_num_threads(int(os.environ['INTRAOP_THREADS']))
    torch.set_num_interop_threads(int(os.environ['INTEROP_THREADS']))

@st.cache_resource
def get_blip_model_processor():
    # print("get model class")
    # model_cls = registry.get_model_class("blip_caption")
    # print("load config")
    # cfg = OmegaConf.load("blip_caption_base_coco.yaml")
    # model_cfg = cfg.model
    # preprocess_cfg = cfg.preprocess
    # print("load preprocess")
    # vis_processors, _ = load_preprocess(preprocess_cfg)
    # print("load model")
    # model = model_cls.from_config(model_cfg)
    # model.eval()

    model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=DEVICE)
    return model, vis_processors["eval"]

@st.cache_resource
def get_pix2struct_model_processor():
    model_id = "models/pix2struct-base"
    model = Pix2StructForConditionalGeneration.from_pretrained(model_id)
    processor = Pix2StructProcessor.from_pretrained(model_id, is_vqa=False)
    return model, processor

@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')

init_pytorch()

NI_MODEL, NI_PROCESSOR = get_blip_model_processor()
SC_MODEL, SC_PROCESSOR = get_pix2struct_model_processor()

def ni_captions(images):
    inputs = [NI_PROCESSOR(image).unsqueeze(0).to(DEVICE) for image in images]
    inputs = torch.stack(inputs).squeeze(1).to(DEVICE)
    out = NI_MODEL.generate({"image": inputs}, num_beams=1, max_length=500)
    return [s.strip() for s in out]

def sc_captions(images):
    inputs = SC_PROCESSOR(images=images, return_tensors="pt").to(DEVICE)
    out = SC_MODEL.generate(**inputs, max_new_tokens=500)
    return [s.strip() for s in SC_PROCESSOR.batch_decode(out, skip_special_tokens=True)]

st.title('Generate captions for your images')
left_column, right_column = st.columns(2)

model = left_column.radio(
    "Model",
    ["Natural Images", "Screenshots (Mobile/Desktop)"],
)

file_names = []
images = []
# TODO: currently 3 in a page - https://github.com/streamlit/streamlit/issues/6454
uploaded_files = left_column.file_uploader("", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
for uploaded_file in uploaded_files:
    file_names.append(uploaded_file.name)
    images.append(Image.open(uploaded_file).convert('RGB'))
if left_column.button('Process', disabled=len(images)==0):
    caption_func = ni_captions if model == "Natural Images" else sc_captions
    progress_text = lambda file_name: f'Operation in progress for {file_name}. Please wait.'
    progress_bar = right_column.progress(0, text=progress_text(file_names[0]))
    captions = []
    for i, (file_name, image) in enumerate(zip(file_names, images)):
        progress_bar.progress(i * 1.0 / len(file_names), text=progress_text(file_name))
        captions.append(caption_func([image])[0])
    progress_bar.empty()
    df = pd.DataFrame({"file_name": file_names, "caption": captions}).set_index('file_name')
    right_column.dataframe(df, width=1200)
    csv = convert_df(df)
    right_column.download_button(
        label="Download caption CSV",
        data=csv,
        file_name='captions.csv',
        mime='text/csv',
    )