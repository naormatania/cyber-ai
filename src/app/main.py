from fastapi import FastAPI, Request
import cyner
from tner import TransformersNER
from nltk.tokenize import WhitespaceTokenizer
from nltk.tokenize.punkt import PunktSentenceTokenizer as pt
from models.consts import LABEL2ID
from transformers import AutoTokenizer
from optimum.bettertransformer import BetterTransformer
import torch
import os

_ = torch.set_grad_enabled(False)
print("default num threads for intraop parallelism: ", torch.get_num_threads())
torch.set_num_threads(int(os.environ['INTRAOP_THREADS'])) # intraop parallelism on CPU
print("current num threads for intraop parallelism: ", torch.get_num_threads())

print("default num threads for interop parallelism: ", torch.get_num_interop_threads())
torch.set_num_interop_threads(int(os.environ['INTEROP_THREADS'])) # interop parallelism on CPU
print("current num threads for interop parallelism: ", torch.get_num_interop_threads())

WARMUP_TEXT = "In late December 2011, CrowdStrike, Inc. received three binary executable files that were suspected of having been involved in a sophisticted attack against a large Fortune 500 company . The files were analyzed to understand first if they were in fact malicious, and the level of sophistication of the samples . The samples were clearly malicious and varied in sophistication . All three samples provided remote access to the attacker, via two Command and Control (C2) servers . One sample is typical of what is commonly referred to as a 'dropper' because its primary purpose is to write a malicious component to disk and connect it to the targeted hosts operating system . The malicious component in this case is what is commonly referred to as a Remote Access Tool (RAT), this RAT is manifested as a Dynamic Link Library (DLL) installed as a service . The second sample analyzed is a dual use tool that can function both as a post exploitation tool used to infect other systems, download additional tools, remove log data, and itself be used as a backdoor . The third sample was a sophisticated implant that in addition to having multiple communication capabilities, and the ability to act as a relay for other infected hosts, utilized a kernel mode driver that can hide aspects of the tool from user-mode tools . This third component is likely used for long-term implantation and intelligence gathering ."

SECUREBERT_NER_MODEL = TransformersNER("models/SecureBERT-NER/", max_length=512, label2id=LABEL2ID)
print("original secure-bert-ner model: ", SECUREBERT_NER_MODEL.model)
SECUREBERT_NER_MODEL.model = BetterTransformer.transform(SECUREBERT_NER_MODEL.model)
print("converted secure-bert-ner model: ", SECUREBERT_NER_MODEL.model)
# Unfortunately compile doesn't work with BetterTransformer and BetterTransformer yields same or better results
#SECUREBERT_NER_MODEL.model = torch.compile(SECUREBERT_NER_MODEL.model, mode="max-autotune-no-cudagraphs")

_ = SECUREBERT_NER_MODEL.predict([WARMUP_TEXT])

CYNER_MODEL = cyner.TransformersNER({'model': 'models/cyner/', 'max_seq_length': 512})
print("original cyner model: ", CYNER_MODEL.classifier.model)
CYNER_MODEL.classifier.model = BetterTransformer.transform(CYNER_MODEL.classifier.model)
print("converted cyner model: ", CYNER_MODEL.classifier.model)
# Unfortunately compile doesn't work with BetterTransformer and BetterTransformer yields same or better results 
#CYNER_MODEL.classifier.model = torch.compile(CYNER_MODEL.classifier.model)

_ = CYNER_MODEL.get_entities_no_split(WARMUP_TEXT)

is_complete_text_cyner = int(os.environ['COMPLETE_TEXT_CYNER'])

def gen_chunk_512(tokenizer, text):
   spans = list(pt().span_tokenize(text))
   num_tokens = 0
   chunk = ""
   for span in spans:
      sub_text = text[span[0]: span[1]]
      tokens = tokenizer.tokenize(sub_text)
      if num_tokens + len(tokens) > 512:
        yield chunk
        chunk = sub_text
        num_tokens = len(tokens)
      elif chunk == "":
        chunk = sub_text
        num_tokens = len(tokens)
      else:
        chunk = " ".join([chunk, sub_text])
        num_tokens = num_tokens + len(tokens)
   yield chunk

def complete_text_cyner(sent, sent_spans, entity):
  start_position = entity.start
  end_position = entity.end

  for span in sent_spans:
    if span[0] < start_position and start_position < span[1]:
       start_position = span[0]
    if span[0] < end_position and end_position < span[1]:
       end_position = span[1]
  
  return sent[start_position:end_position]

app = FastAPI()

@app.post('/ner/cyner/')
async def cyner_endpoint(request: Request):
    data = await request.json()
    text = data.get('text', '')

    if not text:
        return {'error': 'Text to process not found'}
    
    tokenizer = AutoTokenizer.from_pretrained('models/cyner/')
    chunks = gen_chunk_512(tokenizer, text)
    
    entities_tuples = []
    for chunk in chunks:       
        res = CYNER_MODEL.get_entities_no_split(chunk)
        if not is_complete_text_cyner:
           for entity in res:
              entities_tuples.append((entity.entity_type, entity.text))
        elif res:
          sent = res[0].decoded_sent
          sent_spans = WhitespaceTokenizer().span_tokenize(sent)
          for entity in res:
            entity_text = complete_text_cyner(sent, sent_spans, entity)
            entities_tuples.append((entity.entity_type, entity_text))
    
    return {'entities': entities_tuples}

@app.post('/ner/secner/')
async def securebert_ner_endpoint(request: Request):
    data = await request.json()
    text = data.get('text', '')

    if not text:
        return {'error': 'Text to process not found'}
    
    tokenizer = AutoTokenizer.from_pretrained('models/SecureBERT-NER/')
    chunks = gen_chunk_512(tokenizer, text)
    
    entities_tuples = []
    for chunk in chunks:
      res = SECUREBERT_NER_MODEL.predict([chunk])
      last_previous_position = -1
      previous_type = None
      for entity in res['entity_prediction'][0]:
        if last_previous_position == (entity['position'][0]-1) and previous_type == entity['type']:
            entities_tuples[-1] = (entity['type'], ' '.join([entities_tuples[-1][1]]+entity['entity']))
        else:
            entities_tuples.append((entity['type'], ' '.join(entity['entity'])))
        last_previous_position = entity['position'][-1]
        previous_type = entity['type']
    
    return {'entities': entities_tuples}
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000, debug=True)