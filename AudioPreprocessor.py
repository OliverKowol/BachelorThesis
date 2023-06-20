import numpy as np
import torch
import torchaudio
import random
import pandas as pd
import csv

import os
from os import listdir
from os.path import isfile, join


def load_audio_file(file_path):
    """
    Loads an audio file into a tensor
    Args:
        file_path: the audio file path

    Returns: the audio tensor

    """
    audio, sampling_rate = torchaudio.load(file_path)
    return audio


def cut_three_seconds(audio, sampling_rate=16000):
    """
    Cuts or extends an audio to become exactly 2 seconds long
    Args:
        audio: the audio tensor
        sampling_rate: the sampling rate of the audio

    Returns: the two-second audio

    """
    three_seconds = sampling_rate * 3
    current_length = audio.size(dim=1)
    if current_length > three_seconds:  # if audio is longer than 3 seconds
        start = random.randint(0, (current_length - three_seconds))
        three_sec_audio = audio[:, start:start + three_seconds]
    elif current_length < three_seconds:  # if audio is shorter than 3 seconds
        pad_choice = random.randint(0, 1)
        length_diff = three_seconds - current_length
        if pad_choice == 0:  # pad zeros to the beginning
            three_sec_audio = torch.cat((torch.reshape(torch.zeros(length_diff), (1, length_diff)), audio), dim=1)
        else:  # pad zeros to the end
            three_sec_audio = torch.cat((audio, torch.reshape(torch.zeros(length_diff), (1, length_diff))), dim=1)
    else:
        three_sec_audio = audio
    return three_sec_audio


def cal_rms(audio):
    """
    Calculates the root mean square (RMS) of the given audio
    #https://github.com/Sato-Kunihiko/audio-SNR/blob/master/create_mixed_audio_file_with_soundfile.py
    Args:
        audio: the audio tensor

    Returns: the RMS of the audio

    """
    return torch.sqrt(torch.mean(torch.square(audio), axis=-1))


def cal_adjusted_rms(target_rms, snr):
    """
    Calculates the adjusted RMS needed to achieve the given SNR
    #https://github.com/Sato-Kunihiko/audio-SNR/blob/master/create_mixed_audio_file_with_soundfile.py
    Args:
        target_rms: the RMS of the target audio
        snr: the required SNR

    Returns: the adjusted RMS

    """
    a = float(snr) / 20
    noise_rms = target_rms / (10 ** a)
    return noise_rms


def mix_audio_snr(target_audio, noise_audio, snr):
    """
    Adjusts the noise audio RMS to achieve the given SNR
    Args:
        target_audio: the target audio
        noise_audio: the noise audio
        snr: the required SNR

    Returns: the adjusted noise audio

    """
    target_rms = cal_rms(target_audio)
    noise_rms = cal_rms(noise_audio)
    new_noise_rms = cal_adjusted_rms(target_rms, snr)
    new_noise_audio = noise_audio * (new_noise_rms / noise_rms)
    return new_noise_audio


def preprocess_audio(target_audio, utt_audio, noise_audio=None, snr=None, clean=False):
    """
    Preprocesses the audios by making them 2 seconds long, adjusts the RMS of the noise audio, and creates the mix audio
    Args:
        target_audio: the target audio
        utt_audio: the utterance audio
        noise_audio: the noise audio
        snr: the required SNR
        clean: true if clean data required
        utt: true if utterance present

    Returns: the new target audio, the new noise audio, the mix audio, and the SNR

    """
    target_audio = cut_three_seconds(target_audio)
    utt_audio = cut_three_seconds(utt_audio)

    if clean:
        return target_audio, utt_audio

    noise_audio = cut_three_seconds(noise_audio)
    new_noise_audio = mix_audio_snr(target_audio, noise_audio, snr)
    mix_audio = target_audio + new_noise_audio

    # AMPLITUDE
    max_amp = max(np.concatenate((np.array([1]),
                                  np.abs(np.squeeze(mix_audio)),
                                  np.abs(np.squeeze(target_audio)),
                                  np.abs(np.squeeze(noise_audio)))))
    mix_scaling = 1 / max_amp * 0.9

    # MODIFY WITH NEW AMPLITUDE
    target_audio = target_audio * mix_scaling
    utt_audio = utt_audio * mix_scaling
    noise_audio = noise_audio * mix_scaling
    mix_audio = mix_audio * mix_scaling

    return target_audio, utt_audio, noise_audio, mix_audio, snr


def csv_to_array(file_path, num_rows, sv=False, clean=False, is_mix=False):
    """
    Takes a CSV file and extracts its contents to an array
    Args:
        file_path: the CSV file path
        num_rows: the number of rows to extract
        sv: true if preprocessing speaker verification (sv) file
        clean: returns whether sv data will be clean or not
        mix: returns whether audio data will be mixed or not
    Returns: the array of the CSV contents
    """
    df = pd.read_csv(file_path)
    array = []

    if sv:
        for i in range(num_rows):
            if clean:
                row_array = [load_audio_file("Data/wsj0/" + df.loc[i, "Enrollment"] + ".wav"),
                             load_audio_file("Data/wsj0/" + df.loc[i, "Test"] + ".wav")]
            else:
                row_array = [load_audio_file("Data/" + df.loc[i, "Target"]),
                             load_audio_file("Data/" + df.loc[i, "Noise"]),
                             load_audio_file("Data/" + df.loc[i, "Test"]),
                             df.loc[i, "SNR"]]

            row_array.append(df.loc[i, "Key"])
            array.append(row_array)

    else:
        for i in range(num_rows):
            row_array = [load_audio_file("Data/" + df.loc[i, "Target"].split(".")[0] + ".wav"),
                         load_audio_file("Data/" + df.loc[i, "Utterance"].split(".")[0] + ".wav"),
                         load_audio_file("Data/" + df.loc[i, "Noise"].split(".")[0] + ".wav")]
            if is_mix:
                row_array.append(load_audio_file("Data/" + df.loc[i, "Mix"].split(".")[0] + ".wav"))
            row_array.append(df.loc[i, "SNR"])
            array.append(row_array)
    return array


def preprocess_csv(file_path, num_rows, folder_name, sv=False, clean=False, is_mix=False):
    """
    The main preprocess function which combines everything. Converts the CSV to an array, preprocesses the audios,
    and writes the new data to a new CSV file
    Args:
        file_path: the CSV file path
        num_rows: the number of rows to extract
        folder_name: the name of the folder
        sv: true if preprocessing speaker verification (sv) file
        clean: returns whether sv data will be clean or not
        mix: returns whether audio data will be mixed or not
    Returns: None
    """
    dataset = csv_to_array(file_path, num_rows=num_rows, sv=sv, clean=clean, is_mix=is_mix)

    headers = ["Target", "Utterance", "Noise", "Mix", "SNR"] if not sv else ["Enrollment", "Test", "Key"]

    with open("Data/" + folder_name + "/" + folder_name.lower() + "_data_51k.csv", "w",
              newline="") as f:  # changed to 'a'
        writer = csv.writer(f)
        writer.writerow(headers) # commented out when appending
        for i, row in enumerate(dataset):
            if not sv:
                if not is_mix:
                    snr = row[3] * 2
                else:
                    snr = row[4] * 2

                target, utt, noise, mix, snr = preprocess_audio(row[0], row[1], row[2], snr)
                target_path = "Data/" + folder_name + "/Target/target" + str(i) + ".wav"
                torchaudio.save(target_path, target, 16000)
                utt_path = "Data/" + folder_name + "/Utterance/utterance" + str(i) + ".wav"
                torchaudio.save(utt_path, utt, 16000)
                noise_path = "Data/" + folder_name + "/Noise/noise" + str(i) + ".wav"
                torchaudio.save(noise_path, noise, 16000)
                mix_path = "Data/" + folder_name + "/Mix/mix" + str(i) + ".wav"
                torchaudio.save(mix_path, mix, 16000)

                writer.writerow([target_path, utt_path, noise_path, mix_path, snr])

            else:
                if clean:
                    enroll, test = preprocess_audio(target_audio=row[0], utt_audio=row[1], clean=True)
                else:
                    _, test, _, enroll, _ = preprocess_audio(target_audio=row[0], utt_audio=row[2], noise_audio=row[1],
                                                             snr=row[3] * 2)

                enroll_path = "Data/" + folder_name + "/Enrollment/enrollment" + str(
                    i + 10200) + ".wav"  # change to increase count
                torchaudio.save(enroll_path, enroll, 16000)
                test_path = "Data/" + folder_name + "/Test/test" + str(i + 10200) + ".wav"  # change to increase count
                torchaudio.save(test_path, test, 16000)
                key = row[len(row) - 1]

                writer.writerow([enroll_path, test_path, key])


def load_audios(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)

        header = next(reader)
        print(header)
        count = 0

        for i, row in enumerate(reader):
            targ = load_audio_file("Data/" + row[1])
            utt = load_audio_file("Data/" + row[0])
            noise = load_audio_file("Data/" + row[2])

            _, test, _, enroll, _ = preprocess_audio(target_audio=targ, utt_audio=utt, noise_audio=noise,
                                                     snr=float(row[3]) * 2)
            enroll_path = "Data/Test_clean51k/Enrollment/enrollment" + str(i) + ".wav"
            torchaudio.save(enroll_path, enroll, 16000)
            test_path = "Data/Test_clean51k/Test/test" + str(i ) + ".wav"
            torchaudio.save(test_path, test, 16000)

            count += 1

        print("row count:", count)

def load_clean_audios(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)

        header = next(reader)
        print(header)
        count = 0

        for i, row in enumerate(reader):
            enroll = load_audio_file("Data/wsj0/" + row[0] + ".wav")
            test = load_audio_file("Data/wsj0/" + row[1] + ".wav")

            enroll, test = preprocess_audio(target_audio=enroll, utt_audio=test, clean=True)
            enroll_path = "Data/Test_clean51k/Enrollment/enrollment" + str(i) + ".wav"
            torchaudio.save(enroll_path, enroll, 16000)
            test_path = "Data/Test_clean51k/Test/test" + str(i ) + ".wav"
            torchaudio.save(test_path, test, 16000)

            count += 1

        print("row count:", count)



def helper():
    headers = ["Test", "Target", "Noise", "SNR", "Key"]

    with open('Data/xu_test_mix.txt', 'r') as txt_file:
        # Read the data from the text file
        lines = txt_file.readlines()

    # Open the CSV file for writing
    with open('Data/xu_test_mix.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')

        writer.writerow(headers)

        for i, line in enumerate(lines):
            fields = line.split()

            test = fields[0]
            target, snr, noise, _, _ = fields[1].split("_")
            key = 1 if fields[2] == 'target' else 0

            sd_dt_05 = ["00a", "00b", "00c", "00d", "00f", "001", "002", "203", "400", "430", "431", "432"]
            si_dt_05 = ["22g", "22h", "050", "051", "052", "053", "420", "421", "422", "423"]
            si_et_05 = ["440", "441", "442", "443", "444", "445", "446", "447"]

            row = []
            for obj in [test, target, noise]:
                first_three_char = obj[:3]
                if first_three_char in sd_dt_05:
                    prefix = "wsj0/sd_dt_05/" + first_three_char + "/"
                elif first_three_char in si_dt_05:
                    prefix = "wsj0/si_dt_05/" + first_three_char + "/"
                elif first_three_char in si_et_05:
                    prefix = "wsj0/si_et_05/" + first_three_char + "/"
                else:
                    prefix = "wsj0/si_tr_s/" + first_three_char + "/"

                row.append(prefix + obj + ".wav")

            row.append(snr)
            row.append(key)

            writer.writerow(row)


def load_data(file_path, num_rows):
    df = pd.read_csv(file_path)
    array = []
    headers = list(df.columns)

    for i in range(num_rows):
        row_array = []
        for j in headers:
            row_array.append(df.loc[i, j])
        array.append(row_array)
    return array


def get_utt_path(target_path):
    dir_path = get_dir_path(target_path)
    files = create_file_list(dir_path)
    audio = target_path.split(dir_path + "/")[1]

    if files.index(audio) == len(files) - 1:
        utterance_path = dir_path + "/" + files[0]
    else:
        utterance_path = dir_path + "/" + files[files.index(audio) + 1]
    return utterance_path


def create_file_list(dir_path):
    files = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    files.remove('.DS_Store')
    return files


def get_dir_path(path):
    sb = path.split("/")[0]
    for s in range(1, 4):
        sb = sb + "/" + path.split("/")[s]
    return sb


if __name__ == '__main__':
    # DO NOT RUN !!! preprocess_csv("Data/Val5k.csv", num_rows=5000, folder_name="Val5k")
    # DO NOT RUN !!! preprocess_csv("Data/Test3k.csv", num_rows=3000, folder_name="Test3k")
    # DO NOT RUN !!! preprocess_csv("Data/Train20k.csv", num_rows=20000, folder_name="Train20k")
    # DO NOT RUN !!! preprocess_csv("Data/Test_clean51k.csv", num_rows=51000, folder_name="Test_clean51k", sv=True, clean=True)
    # preprocess_csv("Data/test_mix51k_data_51k_2.csv", num_rows=10200, folder_name="Test_mix51k", sv=True) # change file name
    # print(len(pd.read_csv("Data/test_mix51k_data_51k_2.csv")))
    load_clean_audios("Data/Test_clean51k.csv")
    print(len(os.listdir("Data/Test_clean51k/Enrollment")))
    print(len(os.listdir("Data/Test_clean51k/Test")))

    headers = ["Enrollment", "Test", "Key"]
    with open("Data/Test_clean51k/test_clean51k_data_51k.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
    
        for i in range(51000):
            enr = "Data/Test_clean51k/Enrollment/enrollment" + str(i) + ".wav"
            test = "Data/Test_clean51k/Test/test" + str(i) + ".wav"
            key = 1 if i % 17 == 0 else 0
            writer.writerow([enr, test, key])
    print("Done!")


