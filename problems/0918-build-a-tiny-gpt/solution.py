import torch
import torch.nn as nn

class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_layers: int, max_seq_len: int):
        super().__init__()
        # TODO: token_emb, pos_emb, blocks (ModuleList of TransformerEncoderLayer), ln_f, head
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(d_model, nhead=num_heads, dim_feedforward=4*d_model, activation='gelu', batch_first=True, norm_first=True) for _ in range(num_layers)])
        #self.blocks = nn.TransformerEncoder(self.block, num_layers=num_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        pos = torch.arange(idx.shape[1])
        pos_embs = self.pos_emb(pos)
        token_embs = self.token_emb(idx)
        x = pos_embs+token_embs
        masks = nn.Transformer.generate_square_subsequent_mask(idx.shape[1])
        for block in self.blocks:
            x = block(x, src_mask=masks)
        out2 = self.ln_f(x)
        out3 = self.head(out2)
        return out3
        # TODO: embed + pos, run through blocks with causal mask, final LN, project to vocab
        
