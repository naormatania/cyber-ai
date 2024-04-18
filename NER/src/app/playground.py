from fastapi import FastAPI, Request, BackgroundTasks
from tner import TransformersNER
from nltk.tokenize.punkt import PunktSentenceTokenizer as pt
from models.consts import SECNER_LABEL2ID
from transformers import AutoTokenizer
from collections import namedtuple
import pandas as pd
import os
import string
import random

STRIDE_SIZE = 1

tasks_progress = {}
task_entities = {}

Entity = namedtuple("Entity", "type text start_span end_span")

def get_model():
    model = TransformersNER("onnx/SecureBERT_NER/", max_length=512, label2id=SECNER_LABEL2ID)
    warmup_text = "In late December 2011, CrowdStrike, Inc. received three binary executable files that were suspected of having been involved in a sophisticted attack against a large Fortune 500 company . The files were analyzed to understand first if they were in fact malicious, and the level of sophistication of the samples . The samples were clearly malicious and varied in sophistication . All three samples provided remote access to the attacker, via two Command and Control (C2) servers . One sample is typical of what is commonly referred to as a 'dropper' because its primary purpose is to write a malicious component to disk and connect it to the targeted hosts operating system . The malicious component in this case is what is commonly referred to as a Remote Access Tool (RAT), this RAT is manifested as a Dynamic Link Library (DLL) installed as a service . The second sample analyzed is a dual use tool that can function both as a post exploitation tool used to infect other systems, download additional tools, remove log data, and itself be used as a backdoor . The third sample was a sophisticated implant that in addition to having multiple communication capabilities, and the ability to act as a relay for other infected hosts, utilized a kernel mode driver that can hide aspects of the tool from user-mode tools . This third component is likely used for long-term implantation and intelligence gathering ."
    _ = model.predict([warmup_text])
    return model

def get_tokenizer():
   return AutoTokenizer.from_pretrained('onnx/SecureBERT_NER/')

MODEL = get_model()
TOKENIZER = get_tokenizer()

app = FastAPI()

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

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

def chunk_to_entities(model, chunk):
    entities_tuples = []
    res = model.predict([chunk])
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

def process_chunks(task_id, chunks):
    task_entities[task_id] = 0
    entities_tuples = []
    for i, chunk in enumerate(chunks):
        entities_tuples.extend(chunk_to_entities(MODEL, chunk))
        tasks_progress[task_id] = i*1.0/len(chunks)
    tasks_progress[task_id] = 1
    task_entities[task_id] = entities_tuples

@app.post('/start_task/')
async def start_ner_endpoint(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    text = data.get('text', '')

    if not text:
        return {'error': 'Text to process not found'}
    
    chunks = gen_chunk_512(TOKENIZER, text)
    task_id = id_generator()

    background_tasks.add_task(process_chunks, task_id, list(chunks))
    
    return {'task_id': task_id}

@app.post('/get_result/')
async def start_ner_endpoint(request: Request):
    data = await request.json()
    task_id = data.get('task_id', '')

    if not task_id or task_id not in tasks_progress:
        return {'error': 'Task id not found'}
    
    progress = tasks_progress[task_id]
    if progress != 1:
        return {'task_progress': progress}
    else:
        return {'task_progress': tasks_progress.pop(task_id), 'entities': task_entities.pop(task_id)}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.environ['PORT'], debug=True)