import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearEncoder(nn.Module):
    def __init__(self, n_in, n_out, sym=True):
        super(LinearEncoder, self).__init__()

        self.fc = nn.Linear(n_in * 2, n_out)
        self.sym = sym
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Bilinear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def node2edge(self, x):
        # NOTE: Assumes that we have the same graph across all samples.
        # receivers = torch.matmul(rel_rec, x)
        # senders = torch.matmul(rel_send, x)
        # edges = torch.cat([receivers, senders], dim=2)
        N = x.shape[0]
        mask = x.new(1, N).fill_(1)
        node_i = torch.nonzero(mask).repeat(1, N).view(-1, 1)
        node_j = torch.nonzero(mask).repeat(N, 1).view(-1, 1)
        if self.sym:
            triu = (node_i < node_j).squeeze()  # skip loops and symmetric connections
        else:
            triu = (node_i != node_j).squeeze()  # skip loops and symmetric connections
        idx = (node_i * N + node_j)[triu].squeeze()  # linear index
        edges = torch.cat((x[node_i[triu]],
                           x[node_j[triu]]), 1).view(int(torch.sum(triu)), -1)

        return edges, idx

    def edges2matrix(self, x, idx, N):
        edges = x.new(N * N, x.shape[1]).fill_(0)
        edges[idx] = x
        edges = edges.view(N, N, -1)
        return edges

    def forward(self, inputs):
        x = inputs  # N,n_hid
        N = x.shape[0]
        x, idx = self.node2edge(x)  # Eq. 6
        x = self.fc(x)  # Eq. 7: get edge embeddings (N,N,n_hid)
        x = self.edges2matrix(x, idx, N)  # N,N,n_hid
        if self.sym:
            x = x + x.permute(1, 0, 2)
        return x


'''The functions below are adopted from https://github.com/ethanfetaya/NRI'''


class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0., bilinear=False, bnorm=True):
        super(MLP, self).__init__()
        self.bilinear = bilinear
        self.bnorm = bnorm
        if bilinear:
            self.fc1 = nn.Bilinear(n_in, n_in, n_hid)
        else:
            self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        if bnorm:
            self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Bilinear):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = self.bn(inputs)
        return x

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        if self.bilinear:
            x = F.elu(self.fc1(inputs[0], inputs[1]))
            x = x.view(x.size(0), -1)
        else:
            x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        if self.bnorm:
            return self.batch_norm(x)
        else:
            return x


class MLPEncoder(nn.Module):
    def __init__(self, n_in, n_hid, n_out, do_prob=0., factor=True, bilinear=False, n_stages=2, bnorm=True, sym=True):
        super(MLPEncoder, self).__init__()

        self.factor = factor
        self.bilinear = bilinear
        self.n_stages = n_stages
        self.sym = sym
        if self.sym:
            raise NotImplementedError('')

        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob, bnorm=bnorm)
        self.mlp2 = MLP(n_hid * (1 if bilinear else 2), n_hid, n_hid, do_prob, bilinear=bilinear, bnorm=bnorm)

        if n_stages == 2:
            self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
            if self.factor:
                self.mlp4 = MLP(n_hid * (2 if bilinear else 3), n_hid, n_hid, do_prob, bilinear=bilinear, bnorm=False)
                print("Using factor graph MLP encoder.")
            else:
                self.mlp4 = MLP(n_hid * (1 if bilinear else 2), n_hid, n_hid, do_prob, bilinear=bilinear, bnorm=False)
                print("Using MLP encoder.")
        self.fc_out = nn.Linear(n_hid, n_out)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Bilinear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def edge2node(self, x):
        # NOTE: Assumes that we have the same graph across all samples.
        N = x.shape[0]
        device = x.get_device() if x.is_cuda else 'cpu'
        rel_rec = (1 - torch.eye(N, device=device)).unsqueeze(2)  # to sum up all neighbors except for self loops
        incoming = (x * rel_rec).sum(dim=1)  # N,n_hid
        return incoming / N

    def node2edge(self, x):
        # NOTE: Assumes that we have the same graph across all samples.
        N = x.shape[0]
        node_i = torch.arange(N).view(N, 1).repeat(1, N).view(-1, 1)
        node_j = torch.arange(N).view(N, 1).repeat(N, 1).view(-1, 1)
        if self.sym:
            triu = (node_i < node_j).squeeze()  # skip loops and symmetric connections
        else:
            triu = (node_i != node_j).squeeze()  # skip loops
        idx = (node_i * N + node_j)[triu].squeeze()  # linear index
        if self.bilinear:
            edges = (x[node_i[triu]], x[node_j[triu]])
        else:
            edges = torch.cat((x[node_i[triu]],
                               x[node_j[triu]]), 1).view(int(torch.sum(triu)), -1)

        return edges, idx

    def edges2matrix(self, x, idx, N):
        edges = x.new(N * N, x.shape[1]).fill_(0)
        edges[idx] = x
        edges = edges.view(N, N, -1)
        return edges


    def forward(self, inputs, edges=None):
        # Input shape: [num_sims, num_atoms, num_timesteps, num_dims]
        # New shape: [num_sims, num_atoms, num_timesteps*num_dims]
        x = inputs  # N,n_hid
        N = x.shape[0]
        x = self.mlp1(x)  # f_v^1: 2-layer ELU net per node

        x, idx = self.node2edge(x)  # Eq. 6

        x = self.mlp2(x)  # f_e^1: get edge embeddings (N,N,n_hid)

        if self.n_stages == 2:
            x_skip = x  # edge embeddings: N*(N-1)/2, n_hid
            x = self.edges2matrix(x, idx, N)  # N,N,n_hid

            if edges is not None:
                x_skip = self.edges2matrix(x_skip, idx, N)  # N,N,n_hid

                u, v = edges[0, 0].item(), edges[0, 1].item()

                x_skip = torch.cat((x_skip[u, v].view(1, -1), x_skip[v, u].view(1, -1)), dim=0)  # 2,n_hid

                if self.sym:
                    raise NotImplementedError('')

            if self.factor:
                x = self.edge2node(x)  # N,n_hid
                x = x[[u, v], :]  # 2,n_hid
                N = 2
                x = self.mlp3(x)  # f_v^2: 2,n_hid
                x, idx = self.node2edge(x)  # N*(N-1)/2, n_hid
                if self.bilinear:
                    x = (torch.cat((x[0].view(x[0].size(0), -1), x_skip), dim=1),
                         torch.cat((x[1].view(x[1].size(0), -1), x_skip), dim=1))  # Skip connection
                else:
                    x = torch.cat((x, x_skip), dim=1)  # Skip connection
                x = self.mlp4(x)  # N*(N-1)/2, n_hid

                x = self.edges2matrix(x, idx, N)  # N,N,n_hid
            else:
                x = self.mlp3(x)
                x = torch.cat((x, x_skip), dim=1)  # Skip connection
                x = self.mlp4(x)
        else:
            x = self.edges2matrix(x, idx, N)  # N,N,n_hid

        x = self.fc_out(x)  # N,N,n_hid
        if self.sym:
            x = x + x.permute(1, 0, 2)

        return x, idx
