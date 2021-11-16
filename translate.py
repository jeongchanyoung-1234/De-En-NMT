import sys

import torch
from torchtext.data.utils import get_tokenizer

from module.model import Transformer
from module.data_loader import DataLoader


class Config:
    def __init__(self):
        self.batch_size = 128

def translate(sent, dataloader, model_fn='transformer-epoch=29-val_loss=0.0000.ckpt'):
  src = []
  for c in get_tokenizer('moses', language='de')(sent):
    src.append(dataloader.src.vocab.stoi[c.lower().rstrip()])
  src = torch.Tensor(src).unsqueeze(0).long()

  model = Transformer.load_from_checkpoint('/content/drive/MyDrive/data/model/' + model_fn).to('cuda')
  out = model.forward(src.to('cuda'))

  result = []
  for i in out[0]:
    result.append(dataloader.tgt.vocab.itos[i])
  print(' '.join(result[1:-1]))

if __name__ == '__main__':
    dataloader = DataLoader(Config())
    print(translate(' '.join(sys.argv[1:]), dataloader))