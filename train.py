import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from module.model import Transformer
from module.data_loader import DataLoader


def define_argparse():
    p = argparse.ArgumentParser()

    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--epoch', type=int, default=30)
    p.add_argument('--embedding_dim', type=int, default=64)
    p.add_argument('--hidden_size', type=int, default=200)
    p.add_argument('--n_layers', type=int, default=4)
    p.add_argument('--n_head', type=int, default=8)
    p.add_argument('--dropout', type=float, default=.5)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_sharing', action='store_true')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--save_model', action='store_true')
    p.add_argument('--early_stopping', type=int, default=3)

    config = p.parse_args()
    return config

def main(config):
  checkpoint_callback = ModelCheckpoint(
                          dirpath='/content/drive/MyDrive/data/model',
                          filename='transformer-{epoch:2d}-{valid_loss:.4f}',
                          monitor='valid_loss',
                          verbose=True,
                          save_top_k=3,
                        )
  earlystopping_callback = EarlyStopping(
      monitor='valid_loss',
      patience=config.early_stopping,
      verbose=True,
  )
  callbacks = []
  if config.save_model: callbacks.append(checkpoint_callback)
  if config.early_stopping > 0: callbacks.append(earlystopping_callback)
  trainer = pl.Trainer(
      enable_checkpointing=config.save_model,
      callbacks=callbacks,
      gradient_clip_val=5.,
      devices='auto',
      accelerator='gpu',
      auto_select_gpus=True,
      max_epochs=config.epochs)

  dataloader = DataLoader(config)
  src_vocab_size, tgt_vocab_size = len(dataloader.src.vocab), len(dataloader.tgt.vocab)
  train_dataloader, valid_dataloader, test_dataloader= dataloader.get_dataloader()

  model = Transformer(config, src_vocab_size, tgt_vocab_size)
  trainer.fit(model, train_dataloader, valid_dataloader)

if __name__ == '__main__':
    config = define_argparse()
    main(config)