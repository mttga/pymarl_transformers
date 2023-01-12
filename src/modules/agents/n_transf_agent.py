import torch.nn as nn
import torch.nn.functional as F
import torch as th

from ..layer.transformer import Transformer


class TransformerAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(TransformerAgent, self).__init__()

        self.args = args
        self.n_agents   = args.n_agents
        self.n_entities = args.n_entities
        self.feat_dim  = args.obs_entity_feats
        self.emb_dim    = args.emb

        # embedder
        self.feat_embedding = nn.Linear(
            self.feat_dim,
            self.emb_dim
        )

        # transformer block
        self.transformer = Transformer(
            args.emb,
            args.heads,
            args.depth,
            args.ff_hidden_mult,
            args.dropout
        )

        self.q_basic = nn.Linear(args.emb, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return th.zeros(1, self.args.emb).to(self.args.device)

    def forward(self, inputs, hidden_state):

        # prepare the inputs 
        b, a, _ = inputs.size() # batch_size, agents, features
        inputs  = inputs.view(-1, self.n_entities, self.feat_dim)
        hidden_state = hidden_state.view(-1, 1, self.emb_dim)

         # project the embeddings
        embs = self.feat_embedding(inputs)

        # the transformer queries and keys are the input embeddings plus the hidden state
        x = th.cat((hidden_state, embs), 1)

        # get transformer embeddings
        embs = self.transformer.forward(x, x)

        # extract the current hidden state
        h = embs[:, 0:1, :]

        # get the q values
        q = self.q_basic(h)

        return q.view(b, a, -1), h.view(b, a, -1)






