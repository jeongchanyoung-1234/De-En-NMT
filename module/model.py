import torch
import torch.nn as nn
import pytorch_lightning as pl


class Generator(nn.Module) :
    def __init__(self,
                 config,
                 tgt_vocab_size,
                 weight=None) :
        super(Generator, self).__init__()
        self.config = config
        self.linear = nn.Linear(config.embedding_dim, tgt_vocab_size, device=config.device)
        if weight is not None :
            self.linear.weight.data = weight
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x) :
        out = self.softmax(self.linear(x))
        return out


class Transformer(pl.LightningModule) :
    def __init__(self,
                 config,
                 src_vocab_size,
                 tgt_vocab_size,
                 max_length=255,
                 pad_idx=1) :
        super(Transformer, self).__init__()
        self.config = config
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.max_length = max_length
        self.pad_idx = pad_idx

        self.src_embedding = nn.Embedding(self.src_vocab_size, config.embedding_dim, device=config.device)
        self.src_dropout = nn.Dropout(config.dropout)
        self.tgt_embedding = nn.Embedding(self.tgt_vocab_size, config.embedding_dim, device=config.device)
        self.tgt_dropout = nn.Dropout(config.dropout)
        self.pos = self.generate_pos(max_length)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.n_head,
            dim_feedforward=config.hidden_size,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
            device=config.device,
            activation=nn.functional.gelu,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.embedding_dim,
            nhead=config.n_head,
            dim_feedforward=config.hidden_size,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
            device=config.device,
            activation=nn.functional.gelu,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.n_layers)
        self.generator = Generator(config, self.tgt_vocab_size,
                                   self.tgt_embedding.weight.data if config.weight_sharing else None)

        self.save_hyperparameters()

    def generate_pos(self, max_length=20) :
        with torch.no_grad() :
            hidden_size = self.config.embedding_dim
            pos = torch.FloatTensor(max_length, hidden_size).zero_()
            p = torch.arange(0, max_length).unsqueeze(-1).float()
            k = torch.arange(0, hidden_size // 2).unsqueeze(0).float()
            pos[:, 0 : :2] = torch.sin(p / 1e4 ** (k / hidden_size))
            pos[:, 1 : :2] = torch.cos(p / 1e4 ** (k / hidden_size))
        return pos

    def forward(self, src) :
        batch_size = src.size(0)

        src_emb = self.src_embedding(src) * self.config.embedding_dim ** .5
        src_enc = src_emb + self.pos[:src_emb.size(1)].unsqueeze(0).to(self.config.device)
        src_key_padding_mask = (src == self.pad_idx).type(torch.bool).to(self.config.device)
        memory = self.encoder.forward(src_enc, src_key_padding_mask=src_key_padding_mask)

        bos_token = 2
        ys = torch.zeros((batch_size, 1)).long().to(self.config.device) + bos_token
        # |ys| = (bs, length)
        for i in range(self.max_length - 1) :
            tgt_emb = self.tgt_embedding(ys) * self.config.embedding_dim ** .5
            tgt_enc = tgt_emb + self.pos[:tgt_emb.size(1)].unsqueeze(0).to(self.config.device)
            tgt_key_padding_mask = (ys == self.pad_idx).type(torch.bool).to(self.config.device)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(ys.size(1)).to(self.config.device)
            out = self.decoder(
                tgt_enc,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask
            )
            pred = self.generator(out)[:, -1, :].unsqueeze(1).argmax(-1).to(self.config.device)
            ys = torch.cat([ys, pred], dim=1).to(self.config.device)

            if pred == 3 : break

        return ys

    def configure_optimizers(self) :
        optimizer = torch.optim.Adam(self.parameters(), lr=config.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx) :
        src, tgt = train_batch
        tgt_in = tgt[:, :-1]
        with torch.no_grad() :
            src_key_padding_mask = (src == self.pad_idx).type(torch.bool).to(self.config.device)
            tgt_key_padding_mask = (tgt_in == self.pad_idx).type(torch.bool).to(self.config.device)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_in.size(1)).to(self.config.device)

        src_emb = self.src_embedding(src) * self.config.embedding_dim ** .5
        src_enc = src_emb + self.pos[:src_emb.size(1)].unsqueeze(0).to(self.config.device)
        memory = self.encoder.forward(src_enc, src_key_padding_mask=src_key_padding_mask)

        tgt_emb = self.tgt_embedding(tgt_in) * self.config.embedding_dim ** .5
        tgt_enc = tgt_emb + self.pos[:tgt_emb.size(1)].unsqueeze(0).to(self.config.device)
        out = self.decoder(
            tgt_enc,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        y_hat = self.generator(out)
        crit = nn.NLLLoss(ignore_index=self.pad_idx)

        tgt_out = tgt[:, 1 :]
        loss = crit(y_hat.contiguous().view(-1, y_hat.size(-1)), tgt_out.contiguous().view(-1))
        self.log('train_loss', loss.detach(), prog_bar=True)
        self.log('train ppl', torch.exp(loss.detach()), prog_bar=True)
        return {'loss' : loss, 'train ppl' : torch.exp(loss.detach())}

    def validation_step(self, valid_batch, batch_idx) :
        src, tgt = valid_batch
        tgt_in = tgt[:, :-1]

        with torch.no_grad() :
            src_key_padding_mask = (src == self.pad_idx).type(torch.bool).to(self.config.device)
            tgt_key_padding_mask = (tgt_in == self.pad_idx).type(torch.bool).to(self.config.device)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_in.size(1)).to(self.config.device)

        src_emb = self.src_embedding(src) * self.config.embedding_dim ** .5
        src_enc = src_emb + self.pos[:src_emb.size(1), :].unsqueeze(0).to(self.config.device)
        memory = self.encoder.forward(src_enc, src_key_padding_mask=src_key_padding_mask)

        tgt_emb = self.tgt_embedding(tgt_in) * self.config.embedding_dim ** .5
        tgt_enc = tgt_emb + self.pos[:tgt_emb.size(1), :].unsqueeze(0).to(self.config.device)

        out = self.decoder(
            tgt_enc,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        y_hat = self.generator(out)
        crit = nn.NLLLoss(ignore_index=self.pad_idx)

        tgt_out = tgt[:, 1 :]
        loss = crit(y_hat.contiguous().view(-1, y_hat.size(-1)), tgt_out.contiguous().view(-1))
        self.log('valid_loss', loss.detach(), prog_bar=True)
        self.log('valid ppl', torch.exp(loss.detach()), prog_bar=True)
        return {'loss' : loss, 'valid ppl' : torch.exp(loss.detach())}