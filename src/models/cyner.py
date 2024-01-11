import cyner
from nltk.tokenize import WhitespaceTokenizer
import pandas as pd

MODEL128 = cyner.TransformersNER({'model': 'AI4Sec/cyner-xlm-roberta-base', 'max_seq_length': 128})
MODEL512 = cyner.TransformersNER({'model': 'AI4Sec/cyner-xlm-roberta-base', 'max_seq_length': 512})

def fix_text(sent, entity):
  sent_spans = WhitespaceTokenizer().span_tokenize(sent)
  start_position = entity.start
  end_position = entity.end

  for span in sent_spans:
    if span[0] < start_position and start_position < span[1]:
      start_position = span[0]
    if span[0] < end_position and end_position < span[1]:
      end_position = span[1]
  return  sent[start_position:end_position]

def remove_duplicates(df):
  highest_prob = df[['Fixed Text', 'Original Sentence ID', 'Score']].groupby(['Fixed Text', 'Original Sentence ID']).max().reset_index()
  filtered = pd.merge(df, highest_prob, on=['Fixed Text', 'Original Sentence ID', 'Score'], how="left", indicator="Exists")
  filtered = filtered[filtered.Exists == 'both']
  return filtered.drop(columns=["Exists"])

def run_model(ds, model):
    entities_tuples = []
    
    for i in range(len(ds)):
      sent = ds[i]
      res = model.get_entities_no_split(sent)
      for entity in res:
        entity_text = fix_text(entity.decoded_sent, entity)
        entities_tuples.append((entity.entity_type, entity.text, entity_text, entity.confidence, i, entity.sent, entity.decoded_sent))
    
    df = pd.DataFrame(entities_tuples, columns =['Type', 'Text', 'Fixed Text', 'Score', 'Original Sentence ID', 'Original Sentence', 'Decoded Sentence'])
    return remove_duplicates(df)