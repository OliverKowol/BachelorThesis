import torch
import torchaudio
import lightning.pytorch as pl
from torch.utils.data import DataLoader
import pandas as pd
from scipy.io.wavfile import write
import numpy as np


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, num_rows):
        self.data = pd.read_csv(csv_path, nrows=num_rows)

    def __len__(self):
        """
        gets the length of the data
        :return: length
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Auxiliary function for creating batches
        :param index: indexing current row in data
        :return: tensor value of enrollment and utterance
        """
        row = self.data.iloc[index]
        enr_path = row['Enrollment']
        enr, _ = torchaudio.load(enr_path)

        return enr


class TSE_SV(pl.LightningModule):
    def __init__(self, model, num_rows, batch_size):
        super().__init__()
        self.model = model
        self.num_rows = num_rows
        self.batch_size = batch_size

    def forward(self, x):
        return self.model(x)

    def test_step(self, batch, batch_idx):
        """
        executes one test step for the model testing
        :param batch: batch
        :param batch_idx: index of batch
        """
        enr = batch
        enr = enr.cuda()

        # get the estimated output from passing mix through model
        ests = self.forward(enr)

        for i in range(len(ests)):
            est_path = "Data/Test_mix51k/Estimate/estimate_" + str(batch_idx * 4 + i) + ".wav"

            estimate = ests[i].cpu()
            estimate = np.int16(estimate[0, :].numpy() / np.max(np.abs(estimate[0, :].numpy())) * 32767)

            # save estimated audio output
            write(est_path, 16000, estimate.T)

    """
    SETUP LOADER
    """

    def test_dataloader(self):
        test_data = AudioDataset("Data/Test_mix51k/test_mix51k_data_51k.csv", num_rows=self.num_rows)
        test_dl = DataLoader(test_data, batch_size=self.batch_size, shuffle=False, num_workers=6)
        return test_dl
