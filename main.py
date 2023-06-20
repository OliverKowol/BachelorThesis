import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from Lightning import TSE_Model
from SuDORMRF import SuDORMRF

import torch
import time

# to evaluate logged data
# tensorboard --logdir=lightning_logs/

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device {device} is used.\n")


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

    # LOAD PRE-TRAINED WEIGHTS for training from 'scratch'
    pretrained_dict = torch.load('model_weights.pth')
    model_dict = model.state_dict()

    not_init = ['mask_net.1.weight', 'mask_net.1.bias', 'decoder.weight', 'mask_nl_class']

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and k not in not_init}

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    # if CONTINUING TRAINING, comment out previous block and uncomment next line
    # ckpt = "lightning_logs/version_75/checkpoints/baseline-tse-epoch=49-validation_loss=-11.695.ckpt"
    model = TSE_Model(model=model, train_num_rows=20000, val_num_rows=5000, test_num_rows=3000, batch_size=4)

    # LOGGER
    logger = TensorBoardLogger(save_dir="./")
    checkpoint_callback = ModelCheckpoint(filename="baseline-tse-{epoch}-{validation_loss:.3f}",
                                          monitor="validation_loss",
                                          mode="min",
                                          verbose=True,
                                          save_top_k=10
                                          )

    # EARLY STOPPING
    # early_stopping_callback = EarlyStopping(monitor="validation_loss",
    #                                         mode="min",
    #                                         min_delta=0.1,
    #                                         patience=15
    #                                         )

    # TRAINER
    trainer = pl.Trainer(logger=logger,
                         max_epochs=80,
                         gradient_clip_val=5.,
                         callbacks=[checkpoint_callback],  # , early_stopping_callback],
                         accelerator="gpu",
                         devices=1
                         )

    # trainer.fit(model=model, ckpt_path=ckpt)  # if training from checkpoint
    trainer.fit(model=model)  # if training from scratch

    # MISC
    end_time = time.time()

    hours, rem = divmod(end_time - start_time, 3600)
    min, sec = divmod(rem, 60)

    print(
        "Start: {}, End0: {}, Duration: {:0>2}:{:0>2}:{:05.2f}".format(start_time, end_time, int(hours), int(min), sec))
