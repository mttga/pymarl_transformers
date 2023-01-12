import torch.nn as nn
import torch.nn.functional as F
import torch as th

from ..layer.transformer import Transformer


class TransformerAgentSmac(nn.Module):
    def __init__(self, input_shape, args):
        super(TransformerAgentSmac, self).__init__()

        self.args = args
        self.n_agents   = args.n_agents
        self.n_entities = args.n_entities
        self.feat_dim   = args.obs_entity_feats
        self.n_actions  = args.n_actions
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

        # outputs
        self.q_basic  = nn.Linear(args.emb, 6) # moving actions
        self.q_entity = nn.Linear(args.emb, 1) # entity actions (attack)

    def init_hidden(self):
        # make hidden states on same device as model
        return th.zeros(1, self.args.emb).to(self.args.device)

    def forward(self, inputs, hidden_state):

        # process the inputs
        b, a, e = inputs.size()
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

        # hidden output for moving actions
        q = self.q_basic(h)

        # ordered embeddings for enemy-base actions
        q_entity = self.q_entity(embs[:, 1:self.n_entities-self.n_agents+1, :]).view(-1, 1, self.n_entities-self.n_agents)

        # final q
        q = th.cat((q, q_entity), -1)

        return q.view(b, a, -1), h.view(b, a, -1)






