from tner import TransformersNER
from nltk.tokenize.punkt import PunktSentenceTokenizer as pt
from models.consts import LABEL2ID
from transformers import AutoTokenizer
from collections import namedtuple
import pandas as pd
import streamlit as st
import docx

STRIDE_SIZE = 1
BUTTON_DISABLED = True

Entity = namedtuple("Entity", "type text start_span end_span")
st.set_page_config(layout="wide")

@st.cache_resource
def get_model():
    model = TransformersNER("onnx/SecureBERT_NER/", max_length=512, label2id=LABEL2ID)
    warmup_text = "In late December 2011, CrowdStrike, Inc. received three binary executable files that were suspected of having been involved in a sophisticted attack against a large Fortune 500 company . The files were analyzed to understand first if they were in fact malicious, and the level of sophistication of the samples . The samples were clearly malicious and varied in sophistication . All three samples provided remote access to the attacker, via two Command and Control (C2) servers . One sample is typical of what is commonly referred to as a 'dropper' because its primary purpose is to write a malicious component to disk and connect it to the targeted hosts operating system . The malicious component in this case is what is commonly referred to as a Remote Access Tool (RAT), this RAT is manifested as a Dynamic Link Library (DLL) installed as a service . The second sample analyzed is a dual use tool that can function both as a post exploitation tool used to infect other systems, download additional tools, remove log data, and itself be used as a backdoor . The third sample was a sophisticated implant that in addition to having multiple communication capabilities, and the ability to act as a relay for other infected hosts, utilized a kernel mode driver that can hide aspects of the tool from user-mode tools . This third component is likely used for long-term implantation and intelligence gathering ."
    _ = model.predict([warmup_text])
    return model

@st.cache_resource
def get_tokenizer():
   return AutoTokenizer.from_pretrained('models/SecureBERT-NER/')

@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')

MODEL = get_model()
TOKENIZER = get_tokenizer()

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
   df['text'] = df['text'].apply(lambda lst: [lst[i * max_items:(i + 1) * max_items] for i in range((len(lst) + max_items - 1) // max_items )]  )
   return df.explode('text')

def get_docx_text(uploaded_file):
    doc = docx.Document(uploaded_file)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

st.title('Cyber entities extraction')
left_column, right_column = st.columns(2)

uploaded_file = left_column.file_uploader("", type=["txt", "docx"])
if uploaded_file is not None:
    if uploaded_file.type == "text/plain":
        file_text = uploaded_file.getvalue().decode("utf-8")
        BUTTON_DISABLED = False
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        file_text = get_docx_text(uploaded_file)
        BUTTON_DISABLED = False
text = left_column.text_area('Input text:', '', height=500)
if text:
    BUTTON_DISABLED = False
if left_column.button('Process', disabled=BUTTON_DISABLED):
    chunks = list(gen_chunk_512(TOKENIZER, text if text else file_text))
    entities_tuples = []
    progress_text = "Operation in progress. Please wait."
    progress_bar = right_column.progress(0, text=progress_text)
    for i, chunk in enumerate(chunks):
       entities_tuples.extend(chunk_to_entities(chunk))
       progress_bar.progress(i * 1.0 / len(chunks), text=progress_text)
    df = to_showable_pandas(entities_tuples)
    progress_bar.empty()
    right_column.dataframe(df, height=500, width=1200)
    csv = convert_df(df)
    right_column.download_button(
        label="Download entities as CSV",
        data=csv,
        file_name='ner.csv',
        mime='text/csv',
    )