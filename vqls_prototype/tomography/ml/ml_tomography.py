import numpy as np

import pytorch_lightning as pyli

import torch
from IPython.display import display

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchmetrics import Accuracy
import h5py

from .tomography import get_relative_amplitude_sign


def create_dataset(ansatz, size, backend, save_to=None):
    """_summary_

    Args:
        ansatz (_type_): _description_
        size (int): number of datapoint requested
        backend (_type_): _description_
    """
    num_parameters = ansatz.num_parameters
    parameters = 4 * np.pi * np.random.rand(size, num_parameters)
    parameters[:, 2:] = 0

    amplitude_signs = []
    for p in parameters:
        amplitude_signs.append(get_relative_amplitude_sign(ansatz, p, backend))
    np.array(amplitude_signs)

    if save_to is not None:
        with h5py.File(save_to, "w") as h5:
            h5.create_dataset("parameters", data=parameters)
            h5.create_dataset("amplitude_signs", data=np.array(amplitude_signs))

    return parameters, amplitude_signs


class TomographyDataset(Dataset):
    def __init__(self, parameters=None, amplitude_signs=None, load=None):
        self.parameters = parameters
        self.amplitude_signs = amplitude_signs
        self.selected_feature = None

        if load is not None:
            with h5py.File(load, "r") as h5:
                self.parameters = torch.tensor(h5["parameters"][()]).float()
                self.amplitude_signs = torch.tensor(h5["amplitude_signs"][()])

    def select_feature(self, idx):
        self.selected_feature = idx

    def creat_dataset(self, ansatz, size, backend):
        self.parameters, self.amplitude_signs = create_dataset(ansatz, size, backend)

    def __len__(self):
        return len(self.parameters)

    def __getitem__(self, idx):
        if self.selected_feature is None:
            return self.parameters[idx], self.amplitude_signs[idx]
        else:
            idx = int(0.5 * (1 + self.amplitude_signs[idx][self.selected_feature]))
            out = torch.zeros(2)
            out[idx] = 1.0
            return (
                self.parameters[idx],
                out,
            )


class TomographyModel(pyli.LightningModule):
    """Pytorch Model to emulate tomogrpahy data"""

    def __init__(self, size_input, size_output):
        super().__init__()
        self.l1 = torch.nn.Linear(size_input, size_output)

    def forward(self, x):
        return torch.relu(self.l1(x))

    def training_step(self, batch):
        x, y = batch
        x = x.clone().float()
        y = y.clone().float()
        loss = F.cross_entropy(self(x), y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


def train_model(model, dataset, batch_size=64, max_epochs=250):
    train_loader = DataLoader(dataset, batch_size=batch_size)

    # Initialize a trainer
    trainer = pyli.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=max_epochs,
    )

    # Train the model ⚡
    trainer.fit(model, train_loader)
