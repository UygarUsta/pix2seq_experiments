Bbox pix2seq that supports sequence augmentation. Based on good_working_pix2seq's train_pix2seq_bbox script.

### Model Architecture
To upgrade to a full Transformer Encoder-Decoder architecture, the model below can be used. 

> **Note on Padding Mask:** If padding tokens are strictly appended at the end of the sequence , `tgt_key_padding_mask` can be omitted due to causal masking, provided that the loss function ignores the padding index (`ignore_index=PAD_TOKEN`).
```python
class Pix2SeqModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_dim=256,
        nheads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        max_seq_len=200,
        dropout=0.1,
        norm_first=False,          # loss patlarsa True yap (aşağıdaki nota bak)
    ):
        super().__init__()

        # 1. Görsel Omurga (CNN Backbone)
        #resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.enc_proj = nn.Conv2d(2048, hidden_dim, kernel_size=1)  # ResNet50 için 2048 ResNet18 için 512

        # Resim Konum Kodlaması (Learnable Positional Embedding)
        grid_h = IMG_SIZE[0] // 32
        grid_w = IMG_SIZE[1] // 32
        self.img_pos_emb = nn.Parameter(torch.zeros(1, grid_h * grid_w, hidden_dim))
        nn.init.trunc_normal_(self.img_pos_emb, std=0.02)      # randn(std=1.0) yerine

        # 2. Transformer Encoder (Görsel Öz-Dikkat)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(hidden_dim) if norm_first else None,   # pre-LN'de şart
        )

        # 3. Hedef Dizi (Sequence) Modellemesi
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=PAD_TOKEN)
        nn.init.normal_(self.embedding.weight, std=0.02)       # varsayılan N(0,1) yerine
        with torch.no_grad():
            self.embedding.weight[PAD_TOKEN].zero_()

        self.seq_pos_encoding = PositionalEncoding(hidden_dim, max_len=max_seq_len)
        self.emb_dropout = nn.Dropout(dropout)

        # 4. Transformer Decoder (Otoregresif Çözümleme)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
            norm=nn.LayerNorm(hidden_dim) if norm_first else None,
        )
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, images, tgt_seq, tgt_key_padding_mask=None):
        # CNN Özellik Çıkarımı
        features = self.backbone(images)
        features = self.enc_proj(features)

        # (Batch, Dim, H, W) -> (Batch, H*W, Dim)
        src_tokens = features.flatten(2).permute(0, 2, 1)
        src_tokens = src_tokens + self.img_pos_emb

        # Görsel Token'ları Encoder'dan Geçirme (Memory üretimi)
        memory = self.encoder(src_tokens)

        # Hedef Dizi Maskelemesi (Causal Mask)
        tgt_len = tgt_seq.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=images.device)

        # Dizi Gömme
        tgt_emb = self.embedding(tgt_seq)
        tgt_emb = self.seq_pos_encoding(tgt_emb)
        tgt_emb = self.emb_dropout(tgt_emb)

        # Decoder ve Çıkış
        out = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return self.fc_out(out)

```
