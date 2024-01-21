import torch
from torch import cuda
from transformers import BertModel


class BertClass(torch.nn.Module):
    def __init__(self, dropout=0.1):
        super(BertClass, self).__init__()
        self.dropout = dropout
        self.l1 = BertModel.from_pretrained('bert-base-uncased')
        self.l2 = torch.nn.Dropout(dropout)
        self.l3 = torch.nn.Linear(768, 1)

    def forward(self, inputs):
        output_1 = self.l1(**self.unpack_inputs(inputs))['pooler_output']
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output

    def unpack_inputs(self, inputs):
        device = 'cuda' if cuda.is_available() else 'cpu'
        ids = inputs['input_ids'].squeeze().to(device, dtype=torch.long)
        mask = inputs['attention_mask'].squeeze().to(device, dtype=torch.long)
        token_type_ids = inputs['token_type_ids'].squeeze().to(device, dtype=torch.long)
        return {
            'input_ids': ids,
            'attention_mask': mask,
            'token_type_ids': token_type_ids
        }
