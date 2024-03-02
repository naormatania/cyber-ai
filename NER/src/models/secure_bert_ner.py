from tner import TransformersNER
import pandas as pd
from consts import LABEL2ID

MODEL128 = TransformersNER("CyberPeace-Institute/SecureBERT-NER", max_length=128, label2id=LABEL2ID)
MODEL512 = TransformersNER("CyberPeace-Institute/SecureBERT-NER", max_length=512, label2id=LABEL2ID)

def remove_duplicates(df):
    highest_prob = df[['Text', 'Original Sentence ID', 'Score']].groupby(['Text', 'Original Sentence ID']).max().reset_index()
    filtered = pd.merge(df, highest_prob, on=['Text', 'Original Sentence ID', 'Score'], how="left", indicator="Exists")
    filtered = filtered[filtered.Exists == 'both']
    return filtered.drop(columns=["Exists"])

def run_model(ds, model):
    entities_tuples = []

    for i in range(len(ds)):
        sent = ds[i]
        res = model.predict([sent])
        last_previous_position = -1
        previous_type = None
        for entity in res['entity_prediction'][0]:
            if last_previous_position == (entity['position'][0]-1) and previous_type == entity['type']:
                entities_tuples[-1] = (entity['type'], ' '.join([entities_tuples[-1][1]]+entity['entity']), np.mean([entities_tuples[-1][2]]+entity['probability']), i, sent)
            else:
                entities_tuples.append((entity['type'], ' '.join(entity['entity']), np.mean(entity['probability']), i, sent))
            last_previous_position = entity['position'][-1]
            previous_type = entity['type']

    df = pd.DataFrame(entities_tuples, columns =['Type', 'Text', 'Score', 'Original Sentence ID', 'Original Sentence'])
    return remove_duplicates(df)
