import lightning.pytorch as pl
from Lightning import TSE_Model
from SuDORMRF import SuDORMRF

import torch
import time

# to evaluate logged data
# tensorboard --logdir=lightning_logs/

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device {device} is used.\n")


def test_eval(model_specs, version):
    """
    called after testing to retrieve averages of evaluation metrics
    :param model_specs: model information, e.g., step size, decay, ...
    :param version: version folder of trained TSE model
    """
    # Load the .pt file
    filepath = 'lightning_logs/df_test_dict.pt'
    model_state_dict = torch.load(filepath)

    # evaluation metrics
    dicts = ['si-sdr', 'si-sdr-i', 'sdr', 'pesq']

    # writes results to version folder corresponding to TSE model
    with open("lightning_logs/" + version + "/test_results.txt", "w") as f:
        f.write(model_specs)
        for dict_ in dicts:
            d = model_state_dict[dict_]
            values = list(d.values())
            stacked = torch.stack(values)
            average = torch.mean(stacked)

            f.write(dict_ + " average is " + str(average) + "\n")


if __name__ == '__main__':
    start_time = time.time()
    print(start_time)

    # MODEL INITIALISATION
    model = SuDORMRF(out_channels=256,
                     in_channels=512,
                     num_blocks=16,
                     upsampling_depth=5,
                     enc_kernel_size=21,
                     enc_num_basis=512,
                     num_sources=1)

    # LOAD TRAINED VERSION OF MODEL
    version = "version_26_0.2_30"
    epoch = 44
    val_loss = -12.721

    ckpt_path = f"lightning_logs/{version}/checkpoints/baseline-tse-epoch={epoch}-validation_loss={val_loss}.ckpt"
    print(f"Checkpoint: {ckpt_path}\n")

    model_specs = "train data {}, step size {}, decay {}, additional info: {} \n{}, epochs {}, val_loss {}\n\n" \
        .format(20000, 30, 0.3, "-", version, epoch, val_loss)

    # Lightning call
    model = TSE_Model(model=model, train_num_rows=20000, val_num_rows=5000, test_num_rows=3000, batch_size=4)

    # TESTING
    trainer = pl.Trainer()
    trainer.test(model, ckpt_path=ckpt_path)

    # STORE RESULTS
    test_eval(version=version, model_specs=model_specs)

    # MISC
    end_time = time.time()

    hours, rem = divmod(end_time - start_time, 3600)
    min, sec = divmod(rem, 60)

    print(
        "Start: {}, End0: {}, Duration: {:0>2}:{:0>2}:{:05.2f}".format(start_time, end_time, int(hours), int(min), sec))
