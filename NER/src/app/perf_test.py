from tner import TransformersNER
import cyner
from features.malware_text_db import MalwareTextDataset
from features.ner_dataset import TokenizedNERDataset
from models.consts import SECNER_LABEL2ID
import time
import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer
from nltk.tokenize.punkt import PunktSentenceTokenizer as pt

ds = MalwareTextDataset(num_sentences=10)
cyner_tokenized_ds = TokenizedNERDataset(ds, 'AI4Sec/cyner-xlm-roberta-base', 512)
secner_tokenized_ds = TokenizedNERDataset(ds, 'CyberPeace-Institute/SecureBERT-NER', 512)

SECUREBERT_NER_MODEL = TransformersNER("onnx/SecureBERT_NER/", max_length=512, label2id=SECNER_LABEL2ID)
CYNER_MODEL = cyner.TransformersNER({'model': 'onnx/cyner_xlm_roberta_base/', 'max_seq_length': 512})

N_GPU = torch.cuda.device_count()
DEVICE = 'cuda' if N_GPU > 0 else 'cpu'

CYNER_TOKENIZER = AutoTokenizer.from_pretrained('models/cyner/')

class Dataset(torch.utils.data.Dataset):
    """ torch.utils.data.Dataset wrapper converting into tensor """
    float_tensors = ['attention_mask']

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def to_tensor(self, name, data):
        if name in self.float_tensors:
            return torch.tensor(data, dtype=torch.float32)
        return torch.tensor(data, dtype=torch.long)

    def __getitem__(self, idx):
        return {k: self.to_tensor(k, v) for k, v in self.data[idx].items()}

def cyber_library_no_decode(input_list):
    encode_list = CYNER_MODEL.classifier.transforms.encode_plus_all(input_list, max_length=512)
    data_loader = torch.utils.data.DataLoader(Dataset(encode_list), batch_size=len(encode_list))
    encode = list(data_loader)[0]
    encode['attention_mask'] = encode['attention_mask'].type(torch.int64)
    logit = CYNER_MODEL.classifier.model(**{k: v.to(DEVICE) for k, v in encode.items()}, return_dict=True)['logits']
    entities = []
    for n, (s, e) in enumerate(zip(input_list, encode['input_ids'].cpu().tolist())):
        sentence = CYNER_MODEL.classifier.transforms.tokenizer.decode(e, skip_special_tokens=True)
        pred = torch.max(logit[n], dim=-1)[1].cpu().tolist()
        activated = nn.Softmax(dim=-1)(logit[n])
        prob = torch.max(activated, dim=-1)[0].cpu().tolist()
        pred = [CYNER_MODEL.classifier.id_to_label[_p] for _p in pred]
        tag_lists = CYNER_MODEL.classifier.decode_ner_tags(pred, prob)

        _entities = []
        for tag, (start, end) in tag_lists:
            mention = CYNER_MODEL.classifier.transforms.tokenizer.decode(e[start:end], skip_special_tokens=True)
            #Fix 2: mention = mention[1:]
            if not len(mention.strip()): continue

            _ = len(CYNER_MODEL.classifier.transforms.tokenizer.decode(e[:start], skip_special_tokens=True))

            result = {'type': tag, 'position': [start, end], 'mention': mention,
                          'probability': sum(prob[start: end])/(end - start)}
            _entities.append(result)
        entities.append({'entity': _entities, 'sentence': sentence})
    return entities

def full_library_test(input):
    return CYNER_MODEL.get_entities_no_split(input)

def perf_test(model_name, model, tokenized_ds):
    tokenizer_time_arr = []
    model_time_arr = []

    for i in range(len(tokenized_ds)):
        before = time.time()
        encode = tokenized_ds[i]
        elapsed_time = time.time()-before
        tokenizer_time_arr.append(elapsed_time)

        before = time.time()
        _ = model(**{k: [v] for k, v in encode.items()})
        elapsed_time = time.time()-before
        model_time_arr.append(elapsed_time)
    
    print(f"{model_name}: tokenizer_avg_time={np.mean(tokenizer_time_arr)}, tokenizer_std_time={np.std(tokenizer_time_arr)}, tokenizer_median_time={np.median(tokenizer_time_arr)}")
    print(f"{model_name}: model_avg_time={np.mean(model_time_arr)}, model_std_time={np.std(model_time_arr)}, model_median_time={np.median(model_time_arr)}")

def library_test(model_name, model, ds):
    model_time_arr = []

    for i in range(len(ds)):
        text = ds[i]

        before = time.time()
        _ = model([text])
        elapsed_time = time.time()-before
        model_time_arr.append(elapsed_time)
    
    print(f"{model_name}: model_avg_time={np.mean(model_time_arr)}, model_std_time={np.std(model_time_arr)}, model_median_time={np.median(model_time_arr)}")

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

def library_test_no_batch(model_name, model, ds):
    model_time_arr = []
    num_chunks_arr = []

    for i in range(len(ds)):
        text = ds[i]

        chunks = list(gen_chunk_512(CYNER_TOKENIZER, text))
        num_chunks_arr.append(len(chunks))

        before = time.time()
        for chunk in chunks:
            _ = model(chunk)
        elapsed_time = time.time()-before
        model_time_arr.append(elapsed_time)
    
    print(f"{model_name}: model_avg_time={np.mean(model_time_arr)}, model_std_time={np.std(model_time_arr)}, model_median_time={np.median(model_time_arr)}")
    print(f"avg_num_chunks={np.mean(num_chunks_arr)}")

#library_test("cyber_library_no_decode", cyber_library_no_decode, ds)
library_test_no_batch("cyber_library", full_library_test, ds)
#perf_test("cyner", CYNER_MODEL.classifier.model, cyner_tokenized_ds)
#perf_test("secner", SECUREBERT_NER_MODEL.model, secner_tokenized_ds)