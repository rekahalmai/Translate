import torch
import torch.nn as nn 
from lib.layers import EncoderLayer, DecoderLayer
from lib.embed import Embedder, PositionalEncoder
from lib.sublayers import Norm
import copy

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N                                                           # We will have the encoder layer N times
        self.embed = Embedder(vocab_size, d_model)                           # Embeddingn layer
        self.pe = PositionalEncoder(d_model, dropout=dropout)                # PositionalEncoder layer
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)   # Layers of the Enc = N encoder layer
        self.norm = Norm(d_model)                                            # Normalization
    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads, dropout)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)
    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        #print("DECODER")
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output

def get_model(params, src_vocab, trg_vocab):
    
    assert params["d_model"] % params["heads"] == 0
    assert params["dropout"] < 1

    model = Transformer(src_vocab, trg_vocab, params["d_model"], params["n_layers"], params["heads"], params["dropout"])
       
    if params["load_weights"] is not None:
        print("loading pretrained weights...")
        model.load_state_dict(torch.load(f'{params["load_weights"]}/model_weights'))
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) # Fills the tensors with the Glorot initialization
    
    if params["device"] == torch.device('cuda'):
        model = model.cuda()
    
    return model
    
