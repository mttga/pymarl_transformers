import torch.nn as nn
import torch.nn.functional as F
import torch

def mask_(matrices, maskval=0.0, mask_diagonal=True):

    b, h, w = matrices.size()
    indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    matrices[:, indices[0], indices[1]] = maskval

class MultiHeadAttention(nn.Module):
    def __init__(self, emb, heads=8, mask=False):

        super().__init__()

        self.emb = emb
        self.heads = heads
        self.mask = mask

        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)

        self.unifyheads = nn.Linear(heads * emb, emb)

    def forward(self, q, k, mask):
        
        h = self.heads
        # query shape
        b_q, t_q, e_q = q.size()
        # key shape
        b, t_k, e = k.size()

        # check that key and values have the same batch and embedding dim
        assert b == b_q and e == e_q
        
        # get keys, queries, values
        keys = self.tokeys(k).view(b, t_k, h, e)
        values = self.tovalues(k).view(b, t_k, h, e)
        queries = self.toqueries(q).view(b, t_q, h, e)

        # fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t_k, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t_k, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t_q, e)

        # Instead of dividing the dot products by sqrt(e), we scale the keys and values.
        queries = queries / (e ** (1 / 4))
        keys = keys / (e ** (1 / 4))

        # - get dot product of queries and keys
        dot = torch.bmm(queries, keys.transpose(1, 2))
        assert dot.size() == (b * h, t_q, t_k)

        if self.mask:  # mask out the upper half of the dot matrix, excluding the diagonal
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        if mask is not None:
            dot = dot.masked_fill(mask == 0, -1e9)

        # dot as row-wise self-attention probabilities
        dot = F.softmax(dot, dim=2)

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t_q, e)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t_q, h * e)

        return self.unifyheads(out)

class TransformerBlock(nn.Module):

    def __init__(
        self,
        emb,
        heads,
        mask,
        ff_hidden_mult=4,
        dropout=0.0
    ):
        super().__init__()

        self.attention = MultiHeadAttention(emb, heads=heads, mask=mask)
        self.mask = mask

        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )

        self.do = nn.Dropout(dropout)

    def forward(self, q_k_mask):
        q, k, mask = q_k_mask

        attended = self.attention(q, k, mask)

        x = self.norm1(attended + q)

        x = self.do(x)

        fedforward = self.ff(x)

        x = self.norm2(fedforward + x)

        x = self.do(x)

        return x, k, mask


class Transformer(nn.Module):

    def __init__(
        self,
        emb,
        heads,
        depth,
        ff_hidden_mult=4,
        dropout=0.0
    ):
        super().__init__()

        # transformer blocks
        tblocks = []
        for _ in range(depth):
            tblocks.append(
                TransformerBlock(
                    emb=emb,
                    heads=heads,
                    mask=False,
                    ff_hidden_mult=ff_hidden_mult,
                    dropout=dropout
            ))

        self.tblocks = nn.Sequential(*tblocks)

    def forward(self, q, k, mask=None):

        x, k, mask = self.tblocks((q, k, mask))

        return x