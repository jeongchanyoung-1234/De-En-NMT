# De-En NMT
 This is transformer-based NMT, implemented by pytorch-lightning. See more detail below.
 https://arxiv.org/abs/1706.03762
 
## Quick Start
```buildoutcfg
python translate.py [sentence]
```

## Training Parameters
  * batch_size: the size of mini-batches (default=128)
  * epoch: the number of iterations training the whole batches (default=30)
  * embedding_dim: the dimension of embedding vector (default=64)
  * hidden_size: the size of hidden state (default=200)
  * n_layers: the number of transformer layers in both encoder and decoder (default=4)
  * n_head: the number of head in multi-head attention (default=8)
  * dropout: the ratio of nodes which will be dropped (default=.5)
  * lr: the learning rate (default=1e-3)
  * device: device (default='cuda')
  * early_stopping: early stopping patience, not use early stopping if 0 (default=3)  
  * weight_sharing: whether use weightsharing or not (store true)
  * save_model: whether save model or not, best 3 models will be saved (store true)
  