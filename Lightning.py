import torch
import torchaudio
import lightning.pytorch as pl
from torch import optim
from torch.utils.data import DataLoader
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio, signal_distortion_ratio
from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality
import pandas as pd
from collections import OrderedDict
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
        :return: both, tensor value and path of target, noise and mix
        """
        row = self.data.iloc[index]
        target_path = row['Target']
        noise_path = row['Noise']
        mix_path = row['Mix']

        target, _ = torchaudio.load(target_path)
        noise, _ = torchaudio.load(noise_path)
        mix, _ = torchaudio.load(mix_path)

        return target, noise, mix, target_path, noise_path, mix_path


class TSE_Model(pl.LightningModule):
    def __init__(self, model, train_num_rows, val_num_rows, test_num_rows, batch_size):
        super().__init__()
        self.model = model
        self.train_num_rows = train_num_rows
        self.val_num_rows = val_num_rows
        self.test_num_rows = test_num_rows
        self.batch_size = batch_size
        self.dfs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        executes one training step for the model training
        :param batch: batch
        :param batch_idx: index of batch
        :return: calculated loss value SI-SDR
        """
        target, _, mix, _, _, _ = batch
        target = target.cuda()
        mix = mix.cuda()

        # get the estimated output from passing mix through model
        ests = self.forward(mix)

        # calculate SI-SDR loss - quantitative measure
        loss = scale_invariant_signal_distortion_ratio(ests, target).mean(axis=1)
        loss = -torch.sum(loss) / len(ests)
        print("   training batch " + str(batch_idx) + " loss: " + str(loss))
        self.log("training_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        executes one validation step for the model validation
        :param batch: batch
        :param batch_idx: index of batch
        """
        target, _, mix, _, _, _ = batch
        target = target.cuda()
        mix = mix.cuda()

        # get the estimated output from passing mix through model
        ests = self.forward(mix)

        # calculate SI-SDR loss - quantitative measure
        loss = scale_invariant_signal_distortion_ratio(ests, target).mean(axis=1)
        loss = -torch.sum(loss) / len(ests)
        print("   validation batch " + str(batch_idx) + " loss: " + str(loss))

        # calculate PESQ - qualitative measure
        pesq = perceptual_evaluation_speech_quality(ests, target, 16000, "nb").mean(axis=1).mean(axis=0)

        self.log("validation_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("validation_pesq", pesq, on_step=False, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        """
        executes one test step for the model testing
        :param batch: batch
        :param batch_idx: index of batch
        """
        target, _, mix, target_path, noise_path, mix_path = batch
        target = target.cuda()
        mix = mix.cuda()

        # get the estimated output from passing mix through model
        ests = self.forward(mix)

        # create data frame for storing results
        df_test = pd.DataFrame(
            columns=["Mix path", "Target path", "Noise path", "Estimate path", "si-sdr", "si-sdr-i", "sdr", "pesq"])

        for i in range(len(ests)):
            est_path = "Data/Test3k/Ests/est_" + str(batch_idx * self.batch_size + i) + ".wav"
            estimate = ests[i].cpu()
            estimate = np.int16(estimate[0, :].numpy() / np.max(np.abs(estimate[0, :].numpy())) * 32767)

            # save estimated audio output
            write(est_path, 16000, estimate.T)

            # calculate evaluation metrics
            si_sdr = scale_invariant_signal_distortion_ratio(ests[i], target[i])
            si_sdr_i = scale_invariant_signal_distortion_ratio(ests[i], target[i]) \
                       - scale_invariant_signal_distortion_ratio(mix[i], target[i])
            sdr = signal_distortion_ratio(ests[i], target[i])
            pesq = perceptual_evaluation_speech_quality(ests[i], target[i], 16000, "nb")

            # store results in data frame
            df_test.loc[i, "Mix path"] = mix_path[i]
            df_test.loc[i, "Target path"] = target_path[i]
            df_test.loc[i, "Noise path"] = noise_path[i]
            df_test.loc[i, "Estimate path"] = est_path[i]
            df_test.loc[i, "si-sdr"] = si_sdr.cpu()
            df_test.loc[i, "si-sdr-i"] = si_sdr_i.cpu()
            df_test.loc[i, "sdr"] = sdr.cpu()
            df_test.loc[i, "pesq"] = pesq.cpu()
            print(f"metrics, si-sdr: {si_sdr}, si-sdr-i: {si_sdr_i}, sdr: {sdr},  pesq: {pesq} \n")

        df_test.reset_index(inplace=True)
        self.dfs.append(df_test)

    def on_test_epoch_end(self):
        """
        concatenates all sub-data frames into one final data frame and saves it to lightning logs
        """
        df_test = pd.concat(self.dfs, ignore_index=True)
        df_test_dict = df_test.to_dict(into=OrderedDict)
        torch.save(df_test_dict, "lightning_logs/df_test_dict.pt")

    """
    SETUP OPTIMISER AND LOADERS
    """

    def configure_optimizers(self):
        """
        configure optimiser and learning rate scheduler
        """
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.3)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def train_dataloader(self):
        train_data = AudioDataset("Data/Train20k/train20k_data_20k.csv", num_rows=self.train_num_rows)
        train_dl = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=6)
        return train_dl

    def val_dataloader(self):
        val_data = AudioDataset("Data/Val5k/val5k_data_5k.csv", num_rows=self.val_num_rows)
        val_dl = DataLoader(val_data, batch_size=self.batch_size, shuffle=False, num_workers=6)
        return val_dl

    def test_dataloader(self):
        test_data = AudioDataset("Data/Test3k/test_data_3k.csv", num_rows=self.test_num_rows)
        test_dl = DataLoader(test_data, batch_size=self.batch_size, shuffle=False, num_workers=6)
        return test_dl
