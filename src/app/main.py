from fastapi import FastAPI, Request
import cyner
from tner import TransformersNER
from nltk.tokenize import WhitespaceTokenizer
from models.consts import LABEL2ID

SECUREBERT_NER_MODEL = TransformersNER("models/SecureBERT-NER/", max_length=128, label2id=LABEL2ID)
CYNER_MODEL = cyner.TransformersNER({'model': 'models/cyner/', 'max_seq_length': 512})

def fix_text_cyner(sent, entity):
  sent_spans = WhitespaceTokenizer().span_tokenize(sent)
  start_position = entity.start
  end_position = entity.end

  for span in sent_spans:
    if span[0] < start_position and start_position < span[1]:
      start_position = span[0]
    if span[0] < end_position and end_position < span[1]:
      end_position = span[1]
  return  sent[start_position:end_position]

app = FastAPI()

@app.post('/ner/cyner/')
async def cyner_endpoint(request: Request):
    data = await request.json()
    text = data.get('text', '')

    if not text:
        return {'error': 'Text to process not found'}
    
    res = CYNER_MODEL.get_entities_no_split(text)
    entities_tuples = []
    for entity in res:
        entity_text = fix_text_cyner(entity.decoded_sent, entity)
        entities_tuples.append((entity.entity_type, entity_text))
    
    return {'entities': entities_tuples}

@app.post('/ner/secner/')
async def securebert_ner_endpoint(request: Request):
    data = await request.json()
    text = data.get('text', '')

    if not text:
        return {'error': 'Text to process not found'}
    
    res = SECUREBERT_NER_MODEL.predict([text])
    entities_tuples = []

    last_previous_position = -1
    previous_type = None
    for entity in res['entity_prediction'][0]:
        if last_previous_position == (entity['position'][0]-1) and previous_type == entity['type']:
            entities_tuples[-1] = (entity['type'], ' '.join([entities_tuples[-1][1]]+entity['entity']), np.mean([entities_tuples[-1][2]]+entity['probability']), i, sent)
        else:
            entities_tuples.append((entity['type'], ' '.join(entity['entity'])))
        last_previous_position = entity['position'][-1]
        previous_type = entity['type']
    
    return {'entities': entities_tuples}
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000, debug=True)