import torch
from torch import cuda
from transformers import BertModel


class BertClass(torch.nn.Module):
    def __init__(self, dropout=0.1):
        super(BertClass, self).__init__()
        self.device = 'cuda' if cuda.is_available() else 'cpu'
        self.dropout = dropout
        self.l1 = BertModel.from_pretrained('bert-base-uncased')
        self.l2 = torch.nn.Dropout(dropout)
        self.l3 = torch.nn.Linear(768, 1)

    '''
    Definiert die Berechnung, die bei jedem Aufruf durchgeführt wird.

    Siehe: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.forward
    '''
    def forward(self, inputs):
        # Bertbase
        out1 = self.l1(**self.unpack_inputs(inputs))['pooler_output']
        # Dropout
        out2 = self.l2(out1)
        # Linear
        out3 = self.l3(out2)
        return out3

    '''
    Verarbeitet die Eingabedaten so, dass das Modell sie verwenden kann.
    '''
    def unpack_inputs(self, inputs):
        return {
            # Indizes der Eingabesequenz-Tokens im Vokabular, das entspricht den Tokens in der Eingabe
            'input_ids': inputs['input_ids'].squeeze().to(self.device, dtype=torch.long),
            # Wird verwendet, um Aufmerksamkeit auf Padding-Token-Indizes zu vermeiden (0 = maskiert, 1 = nicht maskiert)
            'attention_mask': inputs['attention_mask'].squeeze().to(self.device, dtype=torch.long)
        }
