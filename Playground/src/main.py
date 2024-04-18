from PIL import Image
from models.consts import SECNER_LABEL2NAME
import pandas as pd
import streamlit as st
import docx
import requests
import json
import os
import time

BUTTON_DISABLED = True
NER_ADDRESS = os.environ['NER_ADDRESS']
IMAGE_CAPTIONING_ADDRESS = os.environ['IMAGE_CAPTIONING_ADDRESS']

st.set_page_config(layout="wide")

@st.cache_data
def convert_df(df):
    return df.set_index(['source', 'type'])[['text']].to_csv().encode('utf-8')

@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')

def ner(text, progress_bar, progress_text):
    r = requests.post(f'http://{NER_ADDRESS}/start_task/', data=json.dumps({'text': text}))
    task_id = r.json()['task_id']
    while True:
        r = requests.post(f'http://{NER_ADDRESS}/get_result/', data=json.dumps({'task_id': task_id}))
        resp = r.json()
        if 'task_progress' in resp:
            task_progress = resp['task_progress']
            progress_bar.progress(task_progress, text=progress_text)
            if task_progress == 1:
                return resp['entities']
        time.sleep(1)

def image_captioning(file_name, uploaded_file, type):
    #files = [('files', (os.path.basename(path), open(path, 'rb'), 'image/jpeg'))]
    files = [('files', (file_name, uploaded_file, 'image/jpeg'))]
    r = requests.post(f'http://{IMAGE_CAPTIONING_ADDRESS}/caption/{type}', files=files)
    return r.json()['captions'][0]

def caption_natural_image(file_name, uploaded_file):
    return image_captioning(file_name, uploaded_file, 'natural')

def caption_screenshot_image(file_name, uploaded_file):
    return image_captioning(file_name, uploaded_file, 'screenshot')

def to_showable_pandas(entities_tuples, max_items=5):
   df = pd.DataFrame(entities_tuples, columns=['type', 'text', 'start_pos', 'end_pos'])[['type', 'text']]
   df['text'] = df['text'].apply(lambda text: text.rstrip(",."))
   df = df.groupby(['type']).agg({'text': set})
   df['text'] = df['text'].apply(lambda st: list(st))
   df = df.reset_index()
   df['type'] = df.apply(lambda row: [f'{row["type"]}: {SECNER_LABEL2NAME[row["type"]]}' if i == 0 else row['type'] for i in range((len(row['text']) + max_items - 1) // max_items )], axis=1)
   df['text'] = df['text'].apply(lambda lst: [lst[i * max_items:(i + 1) * max_items] for i in range((len(lst) + max_items - 1) // max_items )])
   df = df.explode(['type', 'text'])
   return df.set_index('type')

def to_downloadable_pandas(entities_tuples):
    df = pd.DataFrame(entities_tuples, columns=['type', 'text', 'start_pos', 'end_pos'])[['type', 'text']]
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
            progress_text = f'Operation in progress for {name}. Please wait.'
            progress_bar = right_column.progress(0, text=progress_text)
            entities_tuples = ner(text, progress_bar, progress_text)
            progress_bar.empty()
            df = to_showable_pandas(entities_tuples)
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
    # TODO: currently 3 in a page - https://github.com/streamlit/streamlit/issues/6454
    uploaded_files = left_column.file_uploader("", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    file_names = [uploaded_file.name for uploaded_file in uploaded_files]
    if left_column.button('Process', disabled=len(file_names)==0):
        progress_text = lambda file_name: f'Operation in progress for {file_name}. Please wait.'
        progress_bar = right_column.progress(0, text=progress_text(file_names[0]))
        captions = []
        for i, (file_name, uploaded_file) in enumerate(zip(file_names, uploaded_files)):
            progress_bar.progress(i * 1.0 / len(file_names), text=progress_text(file_name))
            if model == "Natural Images":
                captions.append(caption_natural_image(file_name, uploaded_file))
            else:
                captions.append(caption_screenshot_image(file_name, uploaded_file))
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