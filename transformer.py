import torch
import torch.nn as nn

#N= batch size, ie the no. of sentence pairs in a batch
#src_len= length of the src sentences, padding= 0
#query_len, key_len, value_len= src/trg sentence length depending on wheather used in encoder or decoder

class MultiHeadAttention(nn.Module):
    def __init__(self, d_emb, heads):
        '''
        d_emb= embedding dim (same as d_model in the paper)
        heads= no. of diff heads in the multi- attention layer
        '''
        super(MultiHeadAttention, self).__init__()
        self.d_emb= d_emb
        self.heads= heads
        assert d_emb % heads== 0
        #dim of each head, to be concatenated later
        self.head_dim= d_emb// heads
        
        self.fc_q= nn.Linear(self.d_emb, self.d_emb)
        self.fc_k= nn.Linear(self.d_emb, self.d_emb)
        self.fc_v= nn.Linear(self.d_emb, self.d_emb)
        
        self.fc_o= nn.Linear(d_emb, d_emb)
    
    def forward(self, query, key, value, mask= None):
        #query shape= N x query_len x d_emb
        N= query.shape[0]
        query_len, key_len, value_len= query.shape[1], key.shape[1], value.shape[1]
        
        #linearly project-> reshape
        query= self.fc_q(query)
        key= self.fc_q(key)
        value= self.fc_q(value)
        query= query.reshape(N, query_len, self.heads, self.head_dim)
        key= key.reshape(N, key_len, self.heads, self.head_dim)
        value= value.reshape(N, value_len, self.heads, self.head_dim)
        
        #query shape= N x query_len x heads x heads_dim
        energy= torch.einsum('nqhd, nkhd-> nhqk', [query, key])
        #energy shape= N x heads x query_len x key_len
        
        if mask is not None:
            energy= energy.masked_fill(mask== 0, float('-inf'))
            
        #apply softmax along the key dim 
        attention_weights= torch.softmax((energy/(self.head_dim)**(1/2)), dim= 3)
        
        #attention_weights shape= N x heads x query_len x key_len
        #value shape= N x value_len x heads x heads_dim
        #out shape= N x query_len x heads x heads_dim
        #out shape= N x query_len x d_emb (concatinating heads)
        out= torch.einsum('nhqk, nvhd-> nqhd', [attention_weights, value]).reshape(N, query_len, -1)
        out= self.fc_o(out)
        return out
        
        
class PositionwiseFeedForwardLayer(nn.Module):
    def __init__(self, d_emb, d_ff):
        '''
        d_ff= dim of the inner layer
        '''
        super(PositionwiseFeedForwardLayer, self).__init__()
        self.features= nn.Sequential(nn.Linear(d_emb, d_ff),
                                     nn.ReLU(),
                                     nn.Linear(d_ff, d_emb))
    
    def forward(self, x):
        out= self.features(x)
        return out
        
class EncoderBlock(nn.Module):
    def __init__(self, d_emb, heads, d_ff, dropout):
        super(EncoderBlock, self).__init__()
        self.self_attention= MultiHeadAttention(d_emb, heads)
        self.norm1= nn.LayerNorm(d_emb)

        self.positionwise_ffn= PositionwiseFeedForwardLayer(d_emb, d_ff)
        self.norm2= nn.LayerNorm(d_emb)
        
        self.dropout= nn.Dropout(p= dropout)
        
    def forward(self, src, src_mask):
        '''
        src_mask= prevents encoder from attending to <pad> tokens
        '''
        out_sa= self.self_attention(src, src, src, src_mask)
        out_sa= self.norm1(src+ self.dropout(out_sa))
        
        out= self.positionwise_ffn(out_sa)
        out= self.norm2(out_sa+ self.dropout(out))
        return out

class Encoder(nn.Module):
    def __init__(self, src_vocab_size, d_emb, max_length, heads, d_ff, dropout, n_blocks):
        super(Encoder, self).__init__()
        self.token_emb= nn.Embedding(src_vocab_size, d_emb)
        #learned positinal embeddings as opposed to the static ones used in the paper
        self.positional_emb= nn.Embedding(max_length, d_emb)
        self.encoder_blocks= nn.ModuleList([EncoderBlock(d_emb, heads, d_ff, dropout) for _ in range(n_blocks)])
        self.dropout= nn.Dropout(p= dropout)
        
    def forward(self, src, src_mask):
        #src shape= N x src_len
        #src_mask shape= N x 1 x 1 x src_len        
        N, src_len= src.shape
        pos= torch.arange(0, src_len).expand(N, src_len)
        
        src= self.dropout(self.token_emb(src)+ self.positional_emb(pos))
        for block in self.encoder_blocks:
            src= block(src, src_mask)
        return src
        
class DecoderBlock(nn.Module):
    def __init__(self, d_emb, heads, d_ff, dropout):
        super(DecoderBlock, self).__init__()
        self.masked_self_attention= MultiHeadAttention(d_emb, heads)
        self.norm1= nn.LayerNorm(d_emb)
        
        self.cross_attention= MultiHeadAttention(d_emb, heads)
        self.norm2= nn.LayerNorm(d_emb)
        
        self.positionwise_ffn= PositionwiseFeedForwardLayer(d_emb, d_ff)
        self.norm3= nn.LayerNorm(d_emb)
        
        self.dropout= nn.Dropout(p= dropout)
        
    def forward(self, trg, trg_mask, encoded_src, src_mask):
        out_msa= self.masked_self_attention(trg, trg, trg, trg_mask)
        out_msa= self.norm1(trg+ self.dropout(out_msa))
        
        out_ca= self.cross_attention(out_msa, encoded_src, encoded_src, src_mask)
        out_ca= self.norm2(out_msa+ self.dropout(out_ca))
        
        out= self.positionwise_ffn(out_ca)
        out= self.norm3(out_ca+ self.dropout(out))
        return out
        
class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, d_emb, max_length, heads, d_ff, dropout, n_blocks):
        super(Decoder, self).__init__()
        self.token_emb= nn.Embedding(trg_vocab_size, d_emb)
        self.positional_emb= nn.Embedding(max_length, d_emb)
        self.decoder_blocks= nn.ModuleList([DecoderBlock(d_emb, heads, d_ff, dropout) for _ in range(n_blocks)])
        self.fc_out = nn.Linear(d_emb, trg_vocab_size)
        self.dropout= nn.Dropout(p= dropout)
        
    def forward(self, trg, trg_mask, encoded_src, src_mask):
        N, trg_len= trg.shape
        pos= torch.arange(0, trg_len).expand(N, trg_len)
        
        trg= self.dropout(self.token_emb(trg)+ self.positional_emb(pos))
        for block in self.decoder_blocks:
            trg= block(trg, trg_mask, encoded_src, src_mask)
        
        out= self.fc_out(trg)
        return out

class Transformer(nn.Module):
    def __init__(self, 
                 src_vocab_size,
                 trg_vocab_size,
                 src_pad_idx,
                 trg_pad_idx,
                 d_emb= 512,
                 heads= 8,
                 d_ff= 2048,
                 n_blocks= 6,
                 dropout= 0.1,
                 max_length= 100):
        
        super(Transformer, self).__init__()
        self.encoder= Encoder(src_vocab_size, d_emb, max_length, heads, d_ff, dropout, n_blocks)
        self.decoder= Decoder(trg_vocab_size, d_emb, max_length, heads, d_ff, dropout, n_blocks)
        self.src_pad_idx= src_pad_idx
        self.trg_pad_idx= trg_pad_idx
        
    def make_src_mask(self, src):
        #src shape= N, src_len
        #src_mask= 0 if <pad> present, else 1
        src_mask= (src!= self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        #src_mask shape= N x 1 x 1 x src_len
        return src_mask
    
    def make_trg_mask(self, trg):
        N, trg_len= trg.shape
        #trg_pad_mask shape= N x 1 x 1 x trg_len
        trg_pad_mask= (trg!= self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        
        #trg_subseq_mask= trg_len x trg_len
        trg_subseq_mask= torch.tril(torch.ones(trg_len, trg_len)).bool()
        
        #trg_mask= N x 1 x trg_len x trg_len
        trg_mask= trg_pad_mask & trg_subseq_mask
        return trg_mask
    
    def forward(self, src, trg):
        src_mask= self.make_src_mask(src)
        trg_mask= self.make_trg_mask(trg)
        encoded_src= self.encoder(src, src_mask)
        out= self.decoder(trg, trg_mask, encoded_src, src_mask)
        return out
    
    
if __name__ == "__main__":
    #depends on the tokenizer
    src_vocab_size = 50
    trg_vocab_size = 50
    
    x= torch.randint(0, 50, (5, 20))
    trg= torch.randint(0, 50, (5, 20))
    
    src_pad_idx = 0
    trg_pad_idx = 0
    
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx)
    #target offset by 1 in addition to masked self attention to avoid info. leak
    out = model(x, trg[:, :-1])
    print(out.shape)
