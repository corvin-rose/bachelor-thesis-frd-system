from django.core.cache import cache
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import transformers
import torch
import os
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

from torch import cuda

device = 'cuda' if cuda.is_available() else 'cpu'


class BERTClass(torch.nn.Module):
    MAX_TEXT_LEN = 512

    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 1)

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.optimizer = torch.optim.Adam(params=self.parameters())

        model_path = os.path.join(os.path.dirname(__file__), '../../resources/model_bert_25ep.pth')
        self.checkpoint = torch.load(model_path)
        self.load_state_dict(self.checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])

    def forward(self, ids, mask, token_type_ids):
        output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)['pooler_output']
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output

    def prepare_text(self, text: str):
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.MAX_TEXT_LEN,
            truncation=True,
            padding='max_length',
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',  # PyTorch Tensoren
        )
        return inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids']

    def classify(self, text: str):
        self.eval()  # Setzen Sie das Modell in den Evaluierungsmodus

        with torch.no_grad():  # Deaktivieren Sie Gradientenberechnung
            input_ids, attention_mask, token_type_ids = self.prepare_text(text)
            output = self.forward(input_ids, attention_mask, token_type_ids)
            output = torch.sigmoid(output).cpu().detach().numpy()
            # Je nach Ihrer Modellarchitektur und Aufgabe müssen Sie möglicherweise die Ausgabe anpassen
            # Zum Beispiel: prediction = outputs.logits.argmax(-1) für Klassifizierungsaufgaben

        return output


def load_bert_model():
    model = cache.get('bert_model')
    if model is None:
        model = BERTClass()
        cache.set('bert_model', model)
    return model
