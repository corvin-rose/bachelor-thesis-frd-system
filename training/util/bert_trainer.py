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

    def loss_fn(self, outputs, targets):
        return torch.nn.BCEWithLogitsLoss()(outputs, targets)

    def train(self, epochs: int):
        loss_list = []
        val_loss_list = []

        start_time = time.time()
        print(f'Start training... lr: {self.get_lr()}, drp: {self.model.dropout}')
        for epoch in range(1, epochs + 1):
            val_loss = self.val_step()
            loss = self.train_step()
            print(f'Epoch: {epoch}, Loss: {np.array(loss).mean()}, Valid. Loss: {val_loss}')
            loss_list.append(loss)
            val_loss_list.append(val_loss)

        loss_list = np.array(loss_list).flatten()
        end_time = time.time()
        training_time = end_time - start_time
        print("Model training time: {:.2f}h".format(training_time / 3600))

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

    def train_step(self):
        loss_list = []
        # Das Modell in den Trainingsmodus versetzen
        self.model.train()
        # Über alle Batches aus dem DataLoader iterieren
        for data in self.train_loader:
            # Daten laden
            inputs, labels = data
            # Verhersage mithilfe von Modell treffen
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

    def val_step(self):
        loss_list = []
        self.model.eval()
        with torch.no_grad():
            for data in self.valid_loader:
                # Daten laden
                inputs, labels = data
                # Verhersage mithilfe von Modell treffen
                outputs = self.model(inputs)
                # Kosten berechnen
                loss = self.loss_fn(outputs, labels.to(self.device, dtype=torch.float))
                loss_list.append(loss.item())
        return np.array(loss_list).mean()

    # https://stackoverflow.com/questions/52660985/pytorch-how-to-get-learning-rate-during-training
    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

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
