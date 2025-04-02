
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from comer.datamodule import CROHMEDatamodule
from comer.lit_comer_swin import LitCoMER 
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import wandb

wandb.init(project="math-recognizer", name="comer-swin")

dm = CROHMEDatamodule(
    zipfile_path='data.zip',
    test_year='2014',
    train_batch_size=2,
    eval_batch_size=2,
    num_workers=2,
    scale_aug=True
)

model = LitCoMER(
    d_model=256,
    nhead=8,
    num_decoder_layers=3,
    dim_feedforward=1024,
    dropout=0.3,
    dc=32,
    cross_coverage=True,
    self_coverage=True,
    beam_size=8,
    max_len=200,
    alpha=1.0,
    early_stopping=False,
    temperature=1.0,
    learning_rate=0.001,
    patience=20,
)

trainer = pl.Trainer(
    devices=1,
    max_epochs=200,
    logger=WandbLogger(),
    default_root_dir='checkpoints',
    callbacks=[
        ModelCheckpoint(
            monitor="val_loss",
            dirpath="checkpoints",
            filename="ComerSwin-{epoch:02d}-{val_loss:.4f}",
            save_top_k=1,
            mode="min",
        ),
        EarlyStopping(monitor="val_loss", patience=5, mode="min"),
    ],
    fast_dev_run=False,
    check_val_every_n_epoch=1,
)

trainer.fit(model, dm)
wandb.finish()
