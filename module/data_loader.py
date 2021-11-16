from torchtext.datasets import Multi30k
from torchtext.legacy.data import Field, TabularDataset, BucketIterator
from torchtext.data.utils import get_tokenizer




def torchtext2csv(dataset_func, path='/content/drive/MyDrive/data/'):
  import pandas as pd
  train, valid, test = Multi30k(language_pair=('de', 'en'))
  columns = ['src', 'tgt']
  train_df = pd.DataFrame(train, columns=columns)
  valid_df = pd.DataFrame(valid, columns=columns)
  test_df = pd.DataFrame(test, columns=columns)
  train_df.to_csv(path + 'train.csv', index=False, encoding='utf-8')
  valid_df.to_csv(path + 'valid.csv', index=False, encoding='utf-8')
  test_df.to_csv(path + 'test.csv', index=False, encoding='utf-8')


class DataLoader:
  def __init__(self,
               config,
               path='/content/drive/MyDrive/data/'):
    super(DataLoader, self).__init__()
    self.config = config
    self.src = Field(
        sequential=True,
        use_vocab=True,
        lower=True,
        tokenize=get_tokenizer('moses', language='de'),
        batch_first=True
    )
    self.tgt = Field(
        sequential=True,
        use_vocab=True,
        lower=True,
        tokenize=get_tokenizer('moses', language='en'),
        is_target=True,
        batch_first=True,
        init_token='<bos>',
        eos_token='<eos>'
    )
    self.train_dataset = TabularDataset(
        path=path + 'train.csv',
        format='csv',
        fields=[('src', self.src), ('tgt', self.tgt)]
    )
    self.valid_dataset = TabularDataset(
        path=path + 'valid.csv',
        format='csv',
        fields=[('src', self.src), ('tgt', self.tgt)]
    )
    self.test_dataset = TabularDataset(
        path=path + 'test.csv',
        format='csv',
        fields=[('src', self.src), ('tgt', self.tgt)]
    )

    self.src.build_vocab(self.train_dataset)
    self.tgt.build_vocab(self.train_dataset)

  def get_dataloader(self):
    train_dataloader = BucketIterator(
        self.train_dataset,
        batch_size=self.config.batch_size,
        sort_key=lambda x: len(x.src) * 255 + len(x.tgt),
        device=self.config.device,
        train=True,
        shuffle=True,
        sort=True,
        sort_within_batch=True
    )

    valid_dataloader = BucketIterator(
        self.valid_dataset,
        batch_size=self.config.batch_size,
        sort_key=lambda x: len(x.src) * 255 + len(x.tgt),
        device=self.config.device,
        train=False,
        shuffle=True,
        sort=True,
        sort_within_batch=True
    )

    test_dataloader = BucketIterator(
        self.test_dataset,
        batch_size=self.config.batch_size,
        sort_key=lambda x: len(x.src) * 255 + len(x.tgt),
        device=self.config.device,
        train=False,
        shuffle=True,
        sort=True,
        sort_within_batch=True
    )

    return train_dataloader, valid_dataloader, test_dataloader