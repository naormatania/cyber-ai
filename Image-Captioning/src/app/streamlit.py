from lavis.models import load_model_and_preprocess, load_preprocess
from PIL import Image
import pandas as pd
import streamlit as st
from lavis.common.registry import registry
from omegaconf import OmegaConf
import torch
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(layout="wide")

@st.cache_resource
def get_model_processor():
    _ = torch.set_grad_enabled(False)
    torch.set_num_threads(int(os.environ['INTRAOP_THREADS']))
    torch.set_num_interop_threads(int(os.environ['INTEROP_THREADS']))

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

@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')

MODEL, PROCESSOR = get_model_processor()

st.title('Generate captions for your images')
left_column, right_column = st.columns(2)

file_names = []
images = []
# TODO: currenly 3 in a page - https://github.com/streamlit/streamlit/issues/6454
uploaded_files = left_column.file_uploader("", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
for uploaded_file in uploaded_files:
    file_names.append(uploaded_file.name)
    images.append(Image.open(uploaded_file))
if left_column.button('Process', disabled=len(images)==0):
    inputs = [PROCESSOR(image).unsqueeze(0).to(DEVICE) for image in images]
    inputs = torch.stack(inputs).squeeze(1).to(DEVICE)
    out = MODEL.generate({"image": inputs}, num_beams=1, max_length=500)
    captions = [s.strip() for s in out]
    df = pd.DataFrame({"file_name": file_names, "caption": captions}).set_index('file_name')
    right_column.dataframe(df, width=1200)
    csv = convert_df(df)
    right_column.download_button(
        label="Download caption CSV",
        data=csv,
        file_name='captions.csv',
        mime='text/csv',
    )