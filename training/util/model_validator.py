import numpy as np
import torch
from torch.utils.data import DataLoader
from training.util.bert_class import BertClass


class ModelValidator:
    def __init__(self, model: BertClass, loader: DataLoader):
        self.model = model
        self.loader = loader

    def calc_accuracy(self):
        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for batch in self.loader:
                inputs, labels = batch
                outputs = self.model(inputs)

                predictions = torch.sigmoid(outputs).cpu().detach().numpy()
                predictions = (np.array(predictions) >= 0.5).astype(int)
                labels = labels.cpu().numpy()

                correct += np.sum(predictions == labels)
                total += len(predictions)

        acc = round(correct / total * 100, 3)
        return acc
