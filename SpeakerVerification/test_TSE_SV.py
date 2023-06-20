from speechbrain.pretrained import EncoderClassifier
from speechbrain.pretrained import SpeakerRecognition

import pandas as pd
import time
import os
import lightning.pytorch as pl

from Lightning_TSE_SV import TSE_SV
from SuDORMRF import SuDORMRF  # change for different TSE models in TSE-SV testing


def duration():
    """
    calculates the total duration of the TSE-SV testing
    """
    hours, rem = divmod(end_time - start_time, 3600)
    min, sec = divmod(rem, 60)
    print(
        "Start: {}, End0: {}, Duration: {:0>2}:{:0>2}:{:05.2f}".format(start_time, end_time, int(hours), int(min), sec))


class TestTSE(object):
    def __init__(self, num_rows):
        self.num_rows = num_rows

        self.model = SuDORMRF(out_channels=256,
                              in_channels=512,
                              num_blocks=16,
                              upsampling_depth=5,
                              enc_kernel_size=21,
                              enc_num_basis=512,
                              num_sources=1)

    def tse(self, ckpt):
        """
        testing trained TSE to create estimates and write to folder
        :param ckpt: checkpoint of trained TSE
        """
        model = TSE_SV(model=self.model, num_rows=self.num_rows, batch_size=4)

        trainer = pl.Trainer()
        trainer.test(model, ckpt_path=ckpt)


class TestSV(object):
    def __init__(self, version, file_path, num_rows):
        # INITIALISE SV MODEL
        self.classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
        
        self.verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                            savedir="pretrained_models/spkrec-ecapa-voxceleb")

        # DATA STORAGE
        self.corr_rej = self.corr_acc = self.false_pos = self.false_neg = 0
        self.scores = []
        self.predictions = []

        self.num_rows = num_rows
        self.file_path = file_path
        self.version = version

    def sv(self):
        """
        performs speaker verification on pre-defined estimates
        """
        print("\nStarting Speaker Verification\n")
        df = pd.read_csv(self.file_path, nrows=self.num_rows)

        est_dir = "Data/Test_mix51k/Estimate"  # alternatives: (1) _clean51k, (2) Enrollment if only SV
        est_audio = os.listdir(est_dir)

        test_dir = "Data/Test_mix51k/Test"  # alternatives: (1) _clean51k
        test_audio = os.listdir(test_dir)

        for i in range(num_rows):
            est = "Data/Test_mix51k/Estimate/" + est_audio[i]  # alternatives: (1) _clean51k, (2) Enrollment if only SV
            test = "Data/Test_mix51k/Test/" + test_audio[i]  # alternatives: (1) _clean51k
            key = df.loc[i, "Key"]

            # calculate score and prediction
            score, prediction = self.verification.verify_files(test, est)

            # store results
            self.scores.append("{:.3f}".format(score.item()))
            self.predictions.append(prediction)

            if prediction == 0 and key == 0:
                self.corr_rej += 1
            elif prediction == 1 and key == 1:
                self.corr_acc += 1
            elif prediction == 1 and key == 0:
                self.false_pos += 1
            elif prediction == 0 and key == 1:
                self.false_neg += 1

            # MISC
            if (i + 1) % (num_rows / 100) == 0:
                print("Process {:.1f}%".format((i + 1) / (num_rows / 100)))
                cleanup()

    def eval(self):
        """
        evaluates SV results
        """
        print("\nEvaluating\n")
        with open("Speaker Verification/Results/TSE_" + version + "_SV_results.txt", "w") as f:  # rename for baselines
            f.write("TSE: {}, SV: {} test samples\n".format(self.version, self.num_rows))

            f.write("Correctly rejected: {}\n".format(self.corr_rej))
            f.write("Correctly accepted: {}\n".format(self.corr_acc))
            f.write("Falsely accepted: {}\n".format(self.false_pos))
            f.write("Falsely rejected: {}\n".format(self.false_neg))

            eer = (self.false_pos + self.false_neg)
            f.write("\nEER%: {:.2f}\n".format(eer))
            f.write("DCF08: {}\n".format("-"))
            f.write("DCF10: {}\n\n".format("-"))

            self.scores.sort()
            f.write("Lowest 0.1% Scores: {}\n".format(self.scores[:51]))
            self.scores.sort(reverse=True)
            f.write("Top 0.1% Scores: {}\n".format(self.scores[:51]))


def cleanup():
    """
    helper function, removes unnecessarily created .wav files
    """
    directory = os.listdir("../Implementation")
    count = 0
    for item in directory:
        if item.endswith(".wav"):
            os.remove(item)
            count += 1
    print("Removed {} .wav files".format(count))


if __name__ == '__main__':
    start_time = time.time()

    os.chdir("..")
    ckpt = "lightning_logs/version_33_bn_0.2_30/checkpoints/baseline-tse-epoch=33-validation_loss=-13.033.ckpt"
    version = ckpt.split("/")[1]
    num_rows = 51000

    test = TestTSE(num_rows)  # comment out for SV testing only
    test.tse(ckpt=ckpt)  # comment out for SV testing only

    file_path = "Data/Test_mix51k/test_mix51k_data_51k.csv"  # alternative: (1) _clean51k
    sv = TestSV(version, file_path, num_rows)
    sv.sv()
    sv.eval()

    # MISC
    end_time = time.time()
    duration()
