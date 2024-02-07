from django.core.cache import cache
import transformers
import torch
import os
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

        # Modell laden
        model_path = os.path.join(os.path.dirname(__file__), '../../resources/model_bert_ep25_lr1e-05_drp0.2.pth')
        self.checkpoint = torch.load(model_path)
        self.load_state_dict(self.checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])

    def forward(self, ids, mask):
        # Bertbase
        out1 = self.l1(ids, attention_mask=mask)['pooler_output']
        # Dropout
        out2 = self.l2(out1)
        # Linear
        out3 = self.l3(out2)
        return out3

    def prepare_text(self, text: str):
        # Eingabetext tokenisieren
        inputs = self.tokenizer.encode_plus(
            text,                           # Eingabe Text
            max_length=self.MAX_TEXT_LEN,   # Maximale Länge für Tokenisierung (wichtig, um padding zu bestimmen)
            truncation=True,                # Zu lange Eingaben werden abgeschnitten
            padding='max_length',           # Padding auf die maximale Länge setzen
            return_tensors='pt'             # In PyTorch Tensors umwandeln
        )
        return inputs['input_ids'], inputs['attention_mask']

    def classify(self, text: str):
        # Das Modell in den Evaluierungsmodus versetzen
        self.eval()
        with torch.no_grad():
            # Text vorbereiten
            input_ids, attention_mask = self.prepare_text(text)
            # Verhersage mithilfe des Modells treffen
            output = self.forward(input_ids, attention_mask)
            # Verhersage als Wahrscheinlichkeit interpretieren
            output = torch.sigmoid(output).cpu().detach().numpy()

        return output


def load_bert_model():
    model = cache.get('bert_model')
    if model is None:
        model = BERTClass()
        cache.set('bert_model', model)
    return model
