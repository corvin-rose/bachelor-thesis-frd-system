import numpy as np
import torch
from torch.utils.data import DataLoader
from training.util.bert_class import BertClass


class ModelValidator:
    def __init__(self, model: BertClass, loader: DataLoader):
        self.model = model
        self.loader = loader

    '''
    Ermittelt die Genauigkeit für ein gegebenes Modell.

    Accuracy = (TP + TN) / (TP + FP + TN + FN)
    '''
    def calc_accuracy(self):
        # TP + TN
        correct = 0
        # TP + FP + TN + FN
        total = 0
        # Das Modell in den Evaluierungsmodus versetzen
        self.model.eval()
        with torch.no_grad():
            # Über alle Batches aus dem DataLoader iterieren
            for batch in self.loader:
                # Daten laden
                inputs, labels = batch
                # Verhersage mithilfe des Modells treffen
                outputs = self.model(inputs)

                # Verhersage in Wahrscheinlichkeit umwandeln
                predictions = torch.sigmoid(outputs).cpu().detach().numpy()
                # Wahrscheinlichkeit interpretieren
                predictions = (np.array(predictions) >= 0.5).astype(int)
                labels = labels.cpu().numpy()
                # Zusammenrechnen
                correct += np.sum(predictions == labels)
                total += len(predictions)

        # Genaugkeit in % berechnen
        acc = round(correct / total * 100, 3)
        return acc
