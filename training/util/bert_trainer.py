import time
import numpy as np
import torch
from torch import cuda
from torch.optim import Adam
from torch.utils.data import DataLoader

from training.util.bert_class import BertClass


class BertTrainer:
    def __init__(self, model: BertClass, optimizer: Adam, train_loader: DataLoader, valid_loader: DataLoader):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = 'cuda' if cuda.is_available() else 'cpu'

    '''
    Die Kostenfunktionfunktion kombiniert eine Sigmoid-Schicht und den BCELoss (Binary Cross-Entropy Loss) 
    in einer einzigen Klasse. Diese Version ist numerisch stabiler als die Verwendung einer einfachen 
    Sigmoid-Funktion, gefolgt von einem BCELoss.

    Siehe: https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
    '''
    def loss_fn(self, outputs, targets):
        return torch.nn.BCEWithLogitsLoss()(outputs, targets)

    '''
    Trainiert ein gegebenes Modell.
    Die epochs sind die Anzahl an Epochen, die trainiert werden.
    Die Methode gibt ein Array der Losses und ein Array der gemittelten validierungs Losses der Epochen zurück.
    '''
    def train(self, epochs: int):
        loss_list = []
        val_loss_list = []

        # Traings-Startzeit ermitteln 
        start_time = time.time()
        print(f'Start training... lr: {self.get_lr()}, drp: {self.model.dropout}')
        
        # Epochen fürs Training durchlaufen
        for epoch in range(1, epochs + 1):
            val_loss = self.val_step()
            loss = self.train_step()
            print(f'Epoch: {epoch}, Loss: {np.array(loss).mean()}, Valid. Loss: {val_loss}')
            loss_list.append(loss)
            val_loss_list.append(val_loss)

        # Loss sammeln
        loss_list = np.array(loss_list).flatten()

        # Trainingszeit ausrechnen
        end_time = time.time()
        training_time = end_time - start_time
        print("Model training time: {:.2f}h".format(training_time / 3600))

        # Modell sichern
        torch.save({
            'epoch': epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss_list,
            'val_loss': val_loss_list,
            'lr': self.get_lr(),
            'drp': self.model.dropout,
            'training_time': training_time
        }, f'model_bert_ep{epochs}_lr{self.get_lr()}_drp{self.model.dropout}.pth')

        return loss_list, val_loss_list

    '''
    Führt einen Trainingsschritt durch: aktualisiert die Gradienten und Gewichtungen. Die Methode 
    gibt ein Array der Kosten des Trainings zurück.
    '''
    def train_step(self):
        loss_list = []
        # Das Modell in den Trainingsmodus versetzen
        self.model.train()
        # Über alle Batches aus dem DataLoader iterieren
        for batch in self.train_loader:
            # Daten laden
            inputs, labels = batch
            # Verhersage mithilfe des Modells treffen
            outputs = self.model(inputs)
            # Gradienten auf null setzen
            self.optimizer.zero_grad()
            # Kosten berechnen
            loss = self.loss_fn(outputs, labels.to(self.device, dtype=torch.float))
            loss_list.append(loss.item())
            # Gradienten mithilfe von Backpropergation berechnen
            loss.backward()
            # Gradienten mithilfe von Optimizer auf Gewichtungen anwenden
            self.optimizer.step()
        return loss_list

    '''
    Führt einen Validierungsschritt durch. Für den aktuellen Zustand des Modells werden die 
    Kosten auf dem Validierungsdatensatz berechnet. Die Methode gibt einen Durchschnitt der 
    Kosten zurück.
    '''
    def val_step(self):
        if self.valid_loader is None:
            return 0

        loss_list = []
        # Das Modell in den Evaluierungsmodus versetzen
        self.model.eval()
        with torch.no_grad():
            for data in self.valid_loader:
                # Daten laden
                inputs, labels = data
                # Verhersage mithilfe des Modells treffen
                outputs = self.model(inputs)
                # Kosten berechnen
                loss = self.loss_fn(outputs, labels.to(self.device, dtype=torch.float))
                loss_list.append(loss.item())
        return np.array(loss_list).mean()

    '''
    Liefert die Lernrate des gegebenen Optimizers.

    Siehe: https://stackoverflow.com/questions/52660985/pytorch-how-to-get-learning-rate-during-training
    '''
    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']


    '''
    Lädt den Checkpoint eines Modells und die damit verbundenen Daten:
    - 'model_state_dict': Modell
    - 'optimizer_state_dict': Optimizer
    - 'loss': Loss Liste
    - 'val_loss': Gemittelte validierungs Loss Liste
    - 'training_time': Trainingszeit
    '''
    def load(self, model: str):
        checkpoint = torch.load(model)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss_list = checkpoint['loss']
        val_loss_list = checkpoint['val_loss']
        training_time = checkpoint['training_time']

        print("Model loaded:", model)
        print("Last loss:", loss_list[-1])
        print("Model training time: {:.2f}h".format(training_time / 3600))

        return loss_list, val_loss_list
