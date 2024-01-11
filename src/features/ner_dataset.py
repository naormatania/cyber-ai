from torch.utils.data import Dataset
from nltk.tokenize.punkt import PunktSentenceTokenizer as pt
import math

class NERDataset(Dataset):
    def __init__(self, df, max_items = None, num_sentences = 1):
        self.num_sentences = num_sentences
        self.sentences = []

        for i in range(len(df)):
            text = df.iloc[i]['text']
            spans = list(pt().span_tokenize(text))
            sents = [text[span[0]: span[1]] for span in spans]
            if max_items is not None:
              num_sents_left = max_items - len(self.sentences)
              self.sentences.extend(sents[:num_sents_left])
              if max_items == len(self.sentences):
                  break
            else:
              self.sentences.extend(sents)

    def __len__(self):
        return math.ceil(len(self.sentences)/self.num_sentences)

    def __getitem__(self, idx):
        sents = self.sentences[idx*self.num_sentences:idx*self.num_sentences+self.num_sentences]
        return ' '.join(sents)