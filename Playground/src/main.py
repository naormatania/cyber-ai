from optimum.onnxruntime import ORTModelForPix2Struct
import torch
from PIL import Image
from lavis.models import load_model_and_preprocess, load_model
from transformers import Pix2StructProcessor
from tner import TransformersNER
from nltk.tokenize.punkt import PunktSentenceTokenizer as pt
from models.consts import SECNER_LABEL2ID, SECNER_LABEL2NAME
from transformers import AutoTokenizer
from collections import namedtuple
import pandas as pd
import streamlit as st
import docx
import os

STRIDE_SIZE = 1
BUTTON_DISABLED = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

Entity = namedtuple("Entity", "type text start_span end_span")
st.set_page_config(layout="wide")

@st.cache_resource
def get_model():
    model = TransformersNER("onnx/SecureBERT_NER/", max_length=512, label2id=SECNER_LABEL2ID)
    warmup_text = "In late December 2011, CrowdStrike, Inc. received three binary executable files that were suspected of having been involved in a sophisticted attack against a large Fortune 500 company . The files were analyzed to understand first if they were in fact malicious, and the level of sophistication of the samples . The samples were clearly malicious and varied in sophistication . All three samples provided remote access to the attacker, via two Command and Control (C2) servers . One sample is typical of what is commonly referred to as a 'dropper' because its primary purpose is to write a malicious component to disk and connect it to the targeted hosts operating system . The malicious component in this case is what is commonly referred to as a Remote Access Tool (RAT), this RAT is manifested as a Dynamic Link Library (DLL) installed as a service . The second sample analyzed is a dual use tool that can function both as a post exploitation tool used to infect other systems, download additional tools, remove log data, and itself be used as a backdoor . The third sample was a sophisticated implant that in addition to having multiple communication capabilities, and the ability to act as a relay for other infected hosts, utilized a kernel mode driver that can hide aspects of the tool from user-mode tools . This third component is likely used for long-term implantation and intelligence gathering ."
    _ = model.predict([warmup_text])
    return model

@st.cache_resource
def get_tokenizer():
   return AutoTokenizer.from_pretrained('onnx/SecureBERT_NER/')

@st.cache_data
def convert_df(df):
    return df.set_index(['source', 'type'])[['text']].to_csv().encode('utf-8')

@st.cache_resource
def init_pytorch():
    _ = torch.set_grad_enabled(False)
    torch.set_num_threads(int(os.environ['INTRAOP_THREADS']))
    torch.set_num_interop_threads(int(os.environ['INTEROP_THREADS']))

@st.cache_resource
def get_blip_model_processor():
    # TODO: change load_model_and_preprocess to load_model (if it is we can just pickle/unpickle the preprocessor): 27.57971429824829s to 15.835256338119507s
    model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=DEVICE)
    return model, vis_processors["eval"]

@st.cache_resource
def get_pix2struct_model_processor():
    model_id = "onnx/pix2struct_base"
    model = ORTModelForPix2Struct.from_pretrained(model_id)
    processor = Pix2StructProcessor.from_pretrained(model_id, is_vqa=False)
    return model, processor

@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')

init_pytorch()

NI_MODEL, NI_PROCESSOR = get_blip_model_processor()
SC_MODEL, SC_PROCESSOR = get_pix2struct_model_processor()
MODEL = get_model()
TOKENIZER = get_tokenizer()

def ni_captions(images):
    inputs = [NI_PROCESSOR(image).unsqueeze(0).to(DEVICE) for image in images]
    inputs = torch.stack(inputs).squeeze(1).to(DEVICE)
    out = NI_MODEL.generate({"image": inputs}, num_beams=1, max_length=500)
    return [s.strip() for s in out]

def sc_captions(images):
    inputs = SC_PROCESSOR(images=images, return_tensors="pt").to(DEVICE)
    out = SC_MODEL.generate(**inputs, max_new_tokens=500)
    return [s.strip() for s in SC_PROCESSOR.batch_decode(out, skip_special_tokens=True)]

def gen_chunk_512(tokenizer, text):
    spans = list(pt().span_tokenize(text))
    num_tokens = 0
    chunk = ""
    previous_stride = []
    previous_stride_num_tokens = []
    for span in spans:
        sub_text = text[span[0]: span[1]]
        tokens = tokenizer.tokenize(sub_text)
        if num_tokens + len(tokens) > 512:
            yield chunk
            chunk = " ".join(previous_stride + [sub_text])
            num_tokens = sum(previous_stride_num_tokens + [len(tokens)])
        elif chunk == "":
            chunk = sub_text
            num_tokens = len(tokens)
        else:
            chunk = " ".join([chunk, sub_text])
            num_tokens = num_tokens + len(tokens)
        previous_stride.append(sub_text)
        previous_stride_num_tokens.append(len(tokens))
        if len(previous_stride) > STRIDE_SIZE:
            previous_stride = previous_stride[1:]
            previous_stride_num_tokens = previous_stride_num_tokens[1:]
    yield chunk

def chunk_to_entities(chunk):
    entities_tuples = []
    res = MODEL.predict([chunk])
    last_previous_position = -1
    previous_type = None
    for entity in res['entity_prediction'][0]:
        if last_previous_position == (entity['position'][0]-1) and previous_type == entity['type']:
            entities_tuples[-1] = Entity(entity['type'], ' '.join([entities_tuples[-1].text]+entity['entity']), entities_tuples[-1].start_span, entity['position'][-1])
        else:
            start_span = entity['position'][0]
            end_span = entity['position'][-1]
            entities_tuples.append(Entity(entity['type'], ' '.join(entity['entity']), start_span, end_span))
        last_previous_position = entity['position'][-1]
        previous_type = entity['type']
    return entities_tuples

def to_showable_pandas(entities_tuples, max_items=5):
   df = pd.DataFrame(entities_tuples)[['type', 'text']]
   df['text'] = df['text'].apply(lambda text: text.rstrip(",."))
   df = df.groupby(['type']).agg({'text': set})
   df['text'] = df['text'].apply(lambda st: list(st))
   df = df.reset_index()
   df['type'] = df.apply(lambda row: [f'{row["type"]}: {SECNER_LABEL2NAME[row["type"]]}' if i == 0 else row['type'] for i in range((len(row['text']) + max_items - 1) // max_items )], axis=1)
   df['text'] = df['text'].apply(lambda lst: [lst[i * max_items:(i + 1) * max_items] for i in range((len(lst) + max_items - 1) // max_items )])
   df = df.explode(['type', 'text'])
   return df.set_index('type')

def to_downloadable_pandas(entities_tuples):
    df = pd.DataFrame(entities_tuples)[['type', 'text']]
    df = df.reset_index()
    df['text'] = df['text'].apply(lambda text: text.rstrip(",."))
    df['type'] = df['type'].apply(lambda type_: f'{type_}: {SECNER_LABEL2NAME[type_]}')
    return df

def get_docx_text(uploaded_file):
    doc = docx.Document(uploaded_file)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

def ner_layout(left_column, right_column):
    texts = {}
    uploaded_files = left_column.file_uploader("", type=["txt", "docx"], accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "text/plain":
            file_text = uploaded_file.getvalue().decode("utf-8")
            texts[uploaded_file.name] = file_text
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            file_text = get_docx_text(uploaded_file)
            texts[uploaded_file.name] = file_text
    text = left_column.text_area('Input text:', '', height=500)
    if text:
        texts['Text'] = text
    if left_column.button('Process', disabled=len(texts)==0):
        all_dfs = []
        for name, text in texts.items():
            chunks = list(gen_chunk_512(TOKENIZER, text if text else file_text))
            entities_tuples = []
            progress_text = f'Operation in progress for {name}. Please wait.'
            progress_bar = right_column.progress(0, text=progress_text)
            for i, chunk in enumerate(chunks):
                entities_tuples.extend(chunk_to_entities(chunk))
                progress_bar.progress(i * 1.0 / len(chunks), text=progress_text)
            df = to_showable_pandas(entities_tuples)
            progress_bar.empty()
            right_column.write(f'{name}:')
            right_column.dataframe(df, width=1200)
            df = to_downloadable_pandas(entities_tuples)
            df['source'] = name
            all_dfs.append(df)
        csv = convert_df(pd.concat(all_dfs))
        right_column.download_button(
            label="Download all entities as CSV",
            data=csv,
            file_name='ner.csv',
            mime='text/csv',
        )

def captioning_layout(left_column, right_column):
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
        progress_text = lambda file_name: f'Operation in progress for {file_name}. Please wait.'
        progress_bar = right_column.progress(0, text=progress_text(file_names[0]))
        captions = []
        for i, (file_name, image) in enumerate(zip(file_names, images)):
            progress_bar.progress(i * 1.0 / len(file_names), text=progress_text(file_name))
            if model == "Natural Images":
                captions.append(ni_captions([image])[0])
            else:
                captions.append(sc_captions([image])[0] + "; " + ni_captions([image])[0])
        progress_bar.empty()
        right_column.image(uploaded_files[0], caption=f'{file_names[0]}: {captions[0]}')
        df = pd.DataFrame({"file_name": file_names, "caption": captions}).set_index('file_name')
        right_column.dataframe(df, width=1200)
        csv = convert_df(df)
        right_column.download_button(
            label="Download caption CSV",
            data=csv,
            file_name='captions.csv',
            mime='text/csv',
        )

st.sidebar.title("App")
app = st.sidebar.radio('', ['Cyber entities extraction', 'Image Captioning'])

st.title(app)
left_column, right_column = st.columns(2)

if app == 'Cyber entities extraction':
    ner_layout(left_column, right_column)
else:
    captioning_layout(left_column, right_column)