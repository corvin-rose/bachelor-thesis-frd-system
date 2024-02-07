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

        model_path = os.path.join(os.path.dirname(__file__), '../../resources/model_bert_ep25_lr1e-05_drp0.2.pth')
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
        self.eval()

        with torch.no_grad():
            input_ids, attention_mask, token_type_ids = self.prepare_text(text)
            output = self.forward(input_ids, attention_mask, token_type_ids)
            output = torch.sigmoid(output).cpu().detach().numpy()

        return output


def load_bert_model():
    model = cache.get('bert_model')
    if model is None:
        model = BERTClass()
        cache.set('bert_model', model)
    return model
