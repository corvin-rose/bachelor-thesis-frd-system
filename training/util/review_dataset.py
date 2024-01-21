import torch

from torch.utils.data import Dataset
from transformers import BertTokenizer

'''
    ReviewDataset Klasse
    
    https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
'''
class ReviewDataset(Dataset):

    def __init__(self, dataframe, max_len: int):
        self.dataframe = dataframe
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_len = max_len

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        text = str(self.dataframe.iloc[index]['text'])
        label = int(self.dataframe.iloc[index]['label_index'])

        # https://huggingface.co/docs/transformers/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode_plus
        inputs = self.tokenizer.encode_plus(
            text,
            # add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            #cls_token=True,
            #return_token_type_ids=True,
            #return_attention_mask=True,
            return_tensors='pt'  # Returns PyTorch tensors
        )
        #ids = inputs['input_ids'].squeeze()  # Remove single-dimensional entries from the shape
        #mask = inputs['attention_mask'].squeeze()
        #token_type_ids = inputs['token_type_ids'].squeeze()
        label_tensor = torch.tensor(label, dtype=torch.float32).unsqueeze(0)

        return inputs, label_tensor

        return {
            'ids': ids,
            'mask': mask,
            'token_type_ids': token_type_ids,
            'label': torch.tensor(label, dtype=torch.float32).unsqueeze(0)  # Convert label to tensor
        }
