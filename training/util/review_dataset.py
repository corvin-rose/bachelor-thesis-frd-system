import torch

from torch.utils.data import Dataset
from transformers import BertTokenizer

'''
ReviewDataset Klasse

Siehe: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
'''
class ReviewDataset(Dataset):

    def __init__(self, dataframe, max_len: int):
        self.dataframe = dataframe
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_len = max_len

    '''
    Ermittelt größe des Datensatzes
    '''
    def __len__(self):
        return len(self.dataframe)

    '''
    Gibt eine tokenisierte Dateninstanz aus dem Datensatz zurück
    '''
    def __getitem__(self, index):
        # Text extrahieren
        text = str(self.dataframe.iloc[index]['text'])
        # Label extrahieren
        label = int(self.dataframe.iloc[index]['label_index'])

        # Siehe: https://huggingface.co/docs/transformers/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode_plus
        inputs = self.tokenizer.encode_plus(
            text,                       # Eingabe Text
            max_length=self.max_len,    # Maximale Länge für Tokenisierung (wichtig, um padding zu bestimmen)
            truncation=True,            # Zu lange Eingaben werden abgeschnitten
            padding='max_length',       # Padding auf die maximale Länge setzen
            return_tensors='pt'         # In PyTorch Tensors umwandeln
        )
        # Label in Tensor umwandeln
        label_tensor = torch.tensor(label, dtype=torch.float32).unsqueeze(0)

        return inputs, label_tensor
