import torch.nn as nn
import torch.nn.functional as F
import torch as th

from ..layer.transformer import Transformer


class UpdetSmac(nn.Module):
    def __init__(self, input_shape, args):
        super(UpdetSmac, self).__init__()

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
        x = th.cat((embs, hidden_state), 1)

        # get transformer embeddings
        embs = self.transformer.forward(x, x)

        # first embedding for moving actions
        q = self.q_basic(embs[:, 0:1, :])

        # ordered embeddings for enemy-base actions
        q_enemies_list = []

        for i in range(self.n_entities-self.n_agents):
            q_enemy = self.q_basic(embs[:, 1 + i, :])
            q_enemy_mean = th.mean(q_enemy, 1, True)
            q_enemies_list.append(q_enemy_mean)

        q_enemies = th.stack(q_enemies_list, dim=1).view(-1, 1, self.n_entities-self.n_agents)

        # final q
        q = th.cat((q, q_enemies), -1)

        # hidden layer
        h = embs[:, -1:, :]

        return q.view(b, a, -1), h.view(b, a, -1)






