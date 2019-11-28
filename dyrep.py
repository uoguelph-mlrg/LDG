import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import *
from utils import *


class DyRep(nn.Module):
    def __init__(self,
                 node_embeddings,
                 A_initial,
                 EVENT_TYPES,
                 N_surv_samples=5,
                 n_hidden=32,
                 bilinear=False,
                 sparse=False,
                 n_rel=1,
                 encoder=None,
                 node_degree_global=None,
                 rnd=None,
                 sym=False,
                 device='cuda'):
        super(DyRep, self).__init__()

        self.opt = True
        self.exp = True
        self.rnd = rnd
        self.bilinear = bilinear
        self.n_hidden = n_hidden
        self.sparse = sparse
        self.encoder = encoder
        self.device = device
        self.N_surv_samples = N_surv_samples
        self.latent_graph = encoder is not None
        self.generate = self.latent_graph

        self.node_degree_global = node_degree_global

        self.N_nodes = A_initial.shape[0]
        if len(A_initial.shape) == 2:
            A_initial = A_initial[:, :, None]

        if self.latent_graph:
            self.n_assoc_types = n_rel
        else:
            self.n_assoc_types = 1

        self.n_relations = self.n_assoc_types + len(EVENT_TYPES)  # 3 communication event types + association event

        self.initialize(node_embeddings, A_initial)

        n_in = 0
        self.W_h = nn.Linear(in_features=n_hidden + n_in, out_features=n_hidden)

        self.W_struct = nn.Linear(n_hidden * self.n_assoc_types, n_hidden)
        self.W_rec = nn.Linear(n_hidden + n_in, n_hidden)
        self.W_t = nn.Linear(4, n_hidden)  # 4 because we want separate parameters for days, hours, minutes, seconds; otherwise (if we just use seconds) it can be a huge number confusing the network

        # Initialize parameters of the intensity rate (edge) prediction functions
        # See https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py
        n_types = 2  # associative and communicative
        d1 = self.n_hidden + (0)
        d2 = self.n_hidden + (0)
        if self.bilinear:
            self.omega = nn.ModuleList([nn.Bilinear(d1, d1, 1), nn.Bilinear(d2, d2, 1)])
        else:
            # d = 2 * self.n_hidden + n_in
            d1 += self.n_hidden
            d2 += self.n_hidden
            self.omega = nn.ModuleList([nn.Linear(d1, 1), nn.Linear(d2, 1)])

        self.psi = nn.Parameter(0.5 * torch.ones(n_types))

        print('omega', self.omega)

        self.train_enc = False
        if encoder is not None:
            if encoder.lower() == 'mlp':
                self.encoder = MLPEncoder(n_in=self.n_hidden, n_hid=self.n_hidden,
                                          n_out=self.n_assoc_types + int(sparse), bilinear=bilinear, n_stages=2,
                                          sym=sym, bnorm=True)
                self.train_enc = True
            elif encoder.lower() == 'mlp1':
                self.encoder = MLPEncoder(n_in=self.n_hidden, n_hid=self.n_hidden,
                                          n_out=self.n_assoc_types + int(sparse), bilinear=bilinear, n_stages=1,
                                          sym=sym, bnorm=True)
                self.train_enc = True
            elif encoder.lower() == 'linear':
                self.encoder = LinearEncoder(n_in=self.n_hidden,
                                             n_out=self.n_assoc_types + int(sparse))
                self.train_enc = True
            elif encoder.lower() == 'rand':
                self.encoder = None
            else:
                raise NotImplementedError(encoder)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Bilinear):
                print('before Xavier', m.weight.data.shape, m.weight.data.min(), m.weight.data.max())
                nn.init.xavier_normal_(m.weight.data)
                print('after Xavier', m.weight.data.shape, m.weight.data.min(), m.weight.data.max())

    def generate_S_from_A(self):
        S = self.A.new(self.N_nodes, self.N_nodes, self.n_assoc_types).fill_(0)
        for rel in range(self.n_assoc_types):
            D = torch.sum(self.A[:, :, rel], dim=1).float()
            for v in torch.nonzero(D):
                u = torch.nonzero(self.A[v, :, rel].squeeze())
                S[v, u, rel] = 1. / D[v]
        self.S = S
        # Check that values in each row of S add up to 1
        for rel in range(self.n_assoc_types):
            S = self.S[:, :, rel]
            assert torch.sum(S[self.A[:, :, rel] == 0]) < 1e-5, torch.sum(S[self.A[:, :, rel] == 0])  # check that S_uv is zero when A_uv is zero


    def initialize(self, node_embeddings, A_initial, keepS=False):
        print('initialize model''s node embeddings and adjacency matrices for %d nodes' % self.N_nodes)
        # Initial embeddings
        if node_embeddings is not None:
            z = np.pad(node_embeddings, ((0, 0), (0, self.n_hidden - node_embeddings.shape[1])), 'constant')
            z = torch.from_numpy(z).float().to(self.device)

        if self.latent_graph:
            print('initial random prediction of A')
            A = torch.zeros(self.N_nodes, self.N_nodes, self.n_assoc_types + int(self.sparse), device=self.device)

            for i in range(self.N_nodes):
                for j in range(i + 1, self.N_nodes):
                    if self.sparse:
                        if self.n_assoc_types == 1:
                            pvals = [0.95, 0.05]
                        elif self.n_assoc_types == 2:
                            pvals = [0.9, 0.05, 0.05]
                        elif self.n_assoc_types == 3:
                            pvals = [0.91, 0.03, 0.03, 0.03]
                        elif self.n_assoc_types == 4:
                            pvals = [0.9, 0.025, 0.025, 0.025, 0.025]
                        else:
                            raise NotImplementedError(self.n_assoc_types)
                        ind = np.nonzero(np.random.multinomial(1, pvals))[0][0]
                    else:
                        ind = np.random.randint(0, self.n_assoc_types, size=1)
                    A[i, j, ind] = 1
                    A[j, i, ind] = 1
            assert torch.sum(torch.isnan(A)) == 0, (torch.sum(torch.isnan(A)), A)
            if self.sparse:
                A = A[:, :, 1:]

        else:
            A = torch.from_numpy(A_initial).float().to(self.device)
            if len(A.shape) == 2:
                A = A.unsqueeze(2)

        # make these variables part of the model
        self.register_buffer('z', z)
        self.register_buffer('A', A)

        if not keepS:
            self.generate_S_from_A()

        self.Lambda_dict = torch.zeros(5000, device=self.device)
        self.time_keys = []

        self.t_p = 0  # global counter of iterations

    def check_S(self):
        for rel in range(self.n_assoc_types):
            rows = torch.nonzero(torch.sum(self.A[:, :, rel], dim=1).float())
            # check that the sum in all rows equal 1
            assert torch.all(torch.abs(torch.sum(self.S[:, :, rel], dim=1)[rows] - 1) < 1e-1), torch.abs(torch.sum(self.S[:, :, rel], dim=1)[rows] - 1)

    def g_fn(self, z_cat, k, edge_type=None, z2=None):
        if self.bilinear:
            B = 1 if len(z_cat.shape) == 1 else z_cat.shape[0]
            if z2 is not None:
                z_cat = z_cat.view(B, self.n_hidden)
                z2 = z2.view(B, self.n_hidden)
            else:
                raise NotImplementedError('')
            g = z_cat.new(len(z_cat), 1).fill_(0)
            idx = k <= 0
            if torch.sum(idx) > 0:
                if edge_type is not None:
                    z_cat1 = torch.cat((z_cat[idx], edge_type.view(B, -1)[idx, :self.n_assoc_types]), dim=1)
                    z21 = torch.cat((z2[idx], edge_type.view(B, -1)[idx, :self.n_assoc_types]), dim=1)
                else:
                    z_cat1 = z_cat[idx]
                    z21 = z2[idx]
                g[idx] = self.omega[0](z_cat1, z21)
            idx = k > 0
            if torch.sum(idx) > 0:
                if edge_type is not None:
                    z_cat1 = torch.cat((z_cat[idx], edge_type.view(B, -1)[idx, self.n_assoc_types:]), dim=1)
                    z21 = torch.cat((z2[idx], edge_type.view(B, -1)[idx, self.n_assoc_types:]), dim=1)
                else:
                    z_cat1 = z_cat[idx]
                    z21 = z2[idx]
                g[idx] = self.omega[1](z_cat1, z21)
        else:
            if z2 is not None:
                z_cat = torch.cat((z_cat, z2), dim=1)
            else:
                raise NotImplementedError('')
            g = z_cat.new(len(z_cat), 1).fill_(0)
            idx = k <= 0
            if torch.sum(idx) > 0:
                if edge_type is not None:
                    z_cat1 = torch.cat((z_cat[idx], edge_type[idx, :self.n_assoc_types]), dim=1)
                else:
                    z_cat1 = z_cat[idx]
                g[idx] = self.omega[0](z_cat1)
            idx = k > 0
            if torch.sum(idx) > 0:
                if edge_type is not None:
                    z_cat1 = torch.cat((z_cat[idx], edge_type[idx, self.n_assoc_types:]), dim=1)
                else:
                    z_cat1 = z_cat[idx]
                g[idx] = self.omega[1](z_cat1)

        g = g.flatten()
        return g


    def intensity_rate_lambda(self, z_u, z_v, k):
        z_u = z_u.view(-1, self.n_hidden).contiguous()
        z_v = z_v.view(-1, self.n_hidden).contiguous()
        edge_type = None
        g = 0.5 * (self.g_fn(z_u, (k > 0).long(), edge_type=edge_type, z2=z_v) +
                   self.g_fn(z_v, (k > 0).long(), edge_type=edge_type, z2=z_u))  # make it symmetric, because most events are symmetric

        psi = self.psi[(k > 0).long()]
        g_psi = torch.clamp(g / (psi + 1e-7), -75, 75)  # to prevent overflow
        Lambda = psi * (torch.log(1 + torch.exp(-g_psi)) + g_psi)
        return Lambda

    def update_node_embed(self, prev_embed, node1, node2, time_delta_uv, k):
        # self.z contains all node embeddings of previous time \bar{t}
        # self.S also corresponds to previous time stamp, because it's not updated yet based on this event

        node_embed = prev_embed

        # compute embeddings for all nodes using the GCN layer, but will be using only nodes u, v
        # it's just not convenient to compute embeddings only for nodes u and v
        # fix that in the future to save computation time

        node_degree = {} # we need degrees to update S
        z_new = prev_embed.clone()  # to allow in place changes while keeping gradients
        h_u_struct = prev_embed.new(2, self.n_hidden, self.n_assoc_types).fill_(0)
        for c, (v, u, delta_t) in enumerate(zip([node1, node2], [node2, node1], time_delta_uv)):  # i is the other node involved in the event
            node_degree[u] = np.zeros(self.n_assoc_types)
            for rel in range(self.n_assoc_types):
                if self.latent_graph:
                    Neighb_u = self.S[u, :, rel] > 1e-7
                else:
                    Neighb_u = self.A[u, :, rel] > 0  # when update embedding for node v, we need neighbors of u and vice versa!
                N_neighb = torch.sum(Neighb_u).item()  # number of neighbors for node u
                node_degree[u][rel] = N_neighb
                if N_neighb > 0:  # node has no neighbors
                    h_prev_i = self.W_h(node_embed[Neighb_u]).view(N_neighb, self.n_hidden)
                    # attention over neighbors
                    q_ui = torch.exp(self.S[u, Neighb_u, rel]).view(N_neighb, 1)
                    q_ui = q_ui / (torch.sum(q_ui) + 1e-7)
                    h_u_struct[c, :, rel] = torch.max(torch.sigmoid(q_ui * h_prev_i), dim=0)[0].view(1, self.n_hidden)

        h1 = self.W_struct(h_u_struct.view(2, self.n_hidden * self.n_assoc_types))

        h2 = self.W_rec(node_embed[[node1, node2], :].view(2, -1))
        h3 = self.W_t(time_delta_uv.float()).view(2, self.n_hidden)

        z_new[[node1, node2], :] = torch.sigmoid(h1 + h2 + h3)


        return node_degree, z_new

    def update_S_A(self, u, v, k, node_degree, lambda_uv_t):

        if self.latent_graph:
            raise ValueError('invalid mode')

        if k <= 0 and not self.latent_graph:  # Association event
            # do not update in case of latent graph
            self.A[u, v, np.abs(k)] = self.A[v, u, np.abs(k)] = 1  # 0 for CloseFriends, k = -1 for the second relation, so it's abs(k) matrix in self.A
        A = self.A
        indices = torch.arange(self.N_nodes, device=self.device)
        for rel in range(self.n_assoc_types):
            if k > 0 and A[u, v, rel] == 0:  # Communication event, no Association exists
                continue  # do not update S and A
            else:
                for j, i in zip([u, v], [v, u]):
                    # i is the "other node involved in the event"
                    try:
                        degree = node_degree[j]
                    except:
                        print(list(node_degree.keys()))
                        raise
                    y = self.S[j, :, rel]
                    # assert torch.sum(torch.isnan(y)) == 0, ('b', j, degree[rel], node_degree_global[rel][j.item()], y)
                    b = 0 if degree[rel] == 0 else 1. / (float(degree[rel]) + 1e-7)
                    if k > 0 and A[u, v, rel] > 0:  # Communication event, Association exists
                        y[i] = b + lambda_uv_t
                    elif k <= 0 and A[u, v, rel] > 0:  # Association event
                        if self.node_degree_global[rel][j] == 0:
                            b_prime = 0
                        else:
                            b_prime = 1. / (float(self.node_degree_global[rel][j]) + 1e-7)
                        x = b_prime - b
                        y[i] = b + lambda_uv_t
                        w = (y != 0) & (indices != int(i))
                        y[w] = y[w] - x
                    y /= (torch.sum(y) + 1e-7)  # normalize
                    self.S[j, :, rel] = y
        return

    def cond_density(self, time_bar, time_cur, u, v):
        N = self.N_nodes
        s = self.Lambda_dict.new(2, N).fill_(0)
        Lambda_sum = torch.cumsum(self.Lambda_dict.flip(0), 0).flip(0)  / len(self.Lambda_dict)
        time_keys_min = self.time_keys[0]
        time_keys_max = self.time_keys[-1]

        indices = []
        l_indices = []
        t_bar_min = torch.min(time_bar[[u, v]]).item()
        if t_bar_min < time_keys_min:
            start_ind_min = 0
        elif t_bar_min > time_keys_max:
            # it means t_bar will always be larger, so there is no history for these nodes
            return s
        else:
            start_ind_min = self.time_keys.index(int(t_bar_min))

        max_pairs = torch.max(torch.cat((time_bar[[u, v]].view(1, 2).expand(N, -1).t().contiguous().view(2 * N, 1),
                                         time_bar.repeat(2, 1)), dim=1), dim=1)[0].view(2, N).long().data.cpu().numpy()  # 2,N

        # compute cond density for all pairs of u and some i, then of v and some i
        c1, c2 = 0, 0
        for c, j in enumerate([u, v]):  # range(i + 1, N):
            for i in range(N):
                if i == j:
                    continue
                # most recent timestamp of either u or v
                t_bar = max_pairs[c, i]
                c2 += 1

                if t_bar < time_keys_min:
                    start_ind = 0  # it means t_bar is beyond the history we kept, so use maximum period saved
                elif t_bar > time_keys_max:
                    continue  # it means t_bar is current event, so there is no history for this pair of nodes
                else:
                    # t_bar is somewhere in between time_keys_min and time_keys_min
                    start_ind = self.time_keys.index(t_bar, start_ind_min)

                indices.append((c, i))
                l_indices.append(start_ind)

        indices = np.array(indices)
        l_indices = np.array(l_indices)
        s[indices[:, 0], indices[:, 1]] = Lambda_sum[l_indices]

        return s

    def edges2matrix(self, x, idx, N):
        edges = x.new(N * N, x.shape[1]).fill_(0)
        edges[idx] = x
        edges = edges.view(N, N, -1)
        return edges

    def generate_S(self, prev_embed, u, v, train_enc=False):
        N = self.N_nodes

        if not train_enc:
            # do not keep any gradients
            with torch.no_grad():
                logits, idx = self.encoder(prev_embed, u, v)
            logits = logits.detach()  # not backpropgenerate_S
        else:
            logits, idx = self.encoder(prev_embed, u, v)

        N = 2
        logits = logits.view(1, N * N, self.n_assoc_types + int(self.sparse))  # N,N,N_assoc  # nn.functional.sigmoid

        edges = gumbel_softmax(logits, tau=0.5, hard=not self.training or not train_enc)  # hard during test time

        if train_enc:
            prob = my_softmax(logits, -1)
            if self.sparse:
                if self.n_assoc_types == 1:
                    log_prior = torch.FloatTensor(np.log(np.array([0.95, 0.05]))).to(self.device)
                    # log_prior = torch.FloatTensor(np.log(np.array([0.9, 0.1]))).to(device)
                elif self.n_assoc_types == 2:
                    log_prior = torch.FloatTensor(np.log(np.array([0.9, 0.05, 0.05]))).to(self.device)
                    # log_prior = torch.FloatTensor(np.log(np.array([0.8, 0.1, 0.1]))).to(device)
                elif self.n_assoc_types == 3:
                    log_prior = torch.FloatTensor(np.log(np.array([0.91, 0.03, 0.03, 0.03]))).to(self.device)
                    # log_prior = torch.FloatTensor(np.log(np.array([0.7, 0.1, 0.1, 0.1]))).to(device)
                elif self.n_assoc_types == 4:
                    log_prior = torch.FloatTensor(np.log(np.array([0.9, 0.025, 0.025, 0.025, 0.025]))).to(self.device)
                else:
                    raise NotImplementedError(self.n_assoc_types)
                log_prior = torch.unsqueeze(log_prior, 0)
                log_prior = torch.unsqueeze(log_prior, 0)
                loss_kl = kl_categorical(prob, log_prior, N)
            else:
                loss_kl = kl_categorical_uniform(prob, N, self.n_assoc_types)  # we want all edge types to have uniform probs
            if torch.isnan(loss_kl):
                print(loss_kl, self.S.min(), self.S.max())
                print(prob)
                raise ValueError()
            reg = [loss_kl]
        else:
            reg = []

        device = edges.get_device() if edges.is_cuda else 'cpu'
        I_neg = 1 - torch.eye(N, device=device).unsqueeze(2)
        edges = edges.view(N, N, -1) * I_neg
        logits = nn.functional.softmax(logits, dim=-1).view(N, N, -1).detach()
        logits = logits * I_neg
        if self.sparse:
            edges = edges[:, :, 1:]
            logits = logits[:, :, 1:]

        return edges, logits, reg

    def forward(self, data):
        u, v, time_delta_uv, event_types, time_bar, time_cur = data[:6]

        B = len(u)
        assert len(event_types) == B, (len(event_types), B)
        N = self.N_nodes

        A_pred, Surv = None, None
        if not self.training:
            A_pred = self.A.new(B, N, N).fill_(0)
            if self.exp:
                Surv = self.A.new(B, N, N).fill_(0)  # survival term

        if self.opt:
            embeddings1, embeddings2, node_degrees = [], [], []
            embeddings_non1, embeddings_non2 = [], []
        else:
            lambda_uv_t, lambda_uv_t_non_events = [], []

        assert torch.min(time_delta_uv) >= 0, ('events must be in chronological order', torch.min(time_delta_uv))

        time_mn = torch.from_numpy(np.array([0, 0, 0, 0])).float().to(self.device).view(1, 1, 4)
        time_sd = torch.from_numpy(np.array([50, 7, 15, 15])).float().to(self.device).view(1, 1, 4)
        time_delta_uv = (time_delta_uv - time_mn) / time_sd

        reg = []

        S_batch = []
        if self.latent_graph:
            if self.encoder is not None and self.t_p == 0:
                print('!!!generate S!!!')
                self.S = self.S / (torch.sum(self.S, dim=1, keepdim=True) + 1e-7)
                self.logits = self.S
                self.A = self.S
                S_batch = [self.S.data.cpu().numpy()]


        z_all = []

        u_all = u.data.cpu().numpy()
        v_all = v.data.cpu().numpy()

        update_attn = not self.latent_graph  # always update if not latent

        for it, k in enumerate(event_types):
            # k = 0: association event (rare)
            # k = 1,2,3: communication event (frequent)

            u_it, v_it = u_all[it], v_all[it]
            z_prev = self.z if it == 0 else z_all[it - 1]

            # 1. Compute intensity rate lambda based on node embeddings at previous time step (Eq. 1)
            if self.opt:
                # store node embeddings, compute lambda and S,A later based on the entire batch
                embeddings1.append(z_prev[u_it])
                embeddings2.append(z_prev[v_it])
            else:
                # accumulate intensity rate of events for this batch based on new embeddings
                lambda_uv_t.append(self.intensity_rate_lambda(z_prev[u_it], z_prev[v_it], torch.zeros(1).long() + k))


            # 2. Update node embeddings
            node_degree, z_new = self.update_node_embed(z_prev, u_it, v_it, time_delta_uv[it], k)  # / 3600.)  # hours
            if self.opt:
                node_degrees.append(node_degree)


            # 3. Update S and A
            if not self.opt and update_attn:
                # we can update S and A based on current pair of nodes even during test time,
                # because S, A are not used in further steps for this iteration
                self.update_S_A(u_it, v_it, k.item(), node_degree, lambda_uv_t[it])  #

            # update most recent degrees of nodes used to update S
            for j in [u_it, v_it]:
                for rel in range(self.n_assoc_types):
                    self.node_degree_global[rel][j] = node_degree[j][rel]

            # Non events loss
            # this is not important for test time, but we still compute these losses for debugging purposes
            # get random nodes except for u_it, v_it
            uv_others = self.rnd.choice(np.delete(np.arange(N), [u_it, v_it]),
                                   size=self.N_surv_samples * 2, replace=False)
            # assert len(np.unique(uv_others)) == len(uv_others), ('nodes must be unique', uv_others)
            for q in range(self.N_surv_samples):
                assert u_it != uv_others[q], (u_it, uv_others[q])
                assert v_it != uv_others[self.N_surv_samples + q], (v_it, uv_others[self.N_surv_samples + q])
                if self.opt:
                    embeddings_non1.extend([z_prev[u_it], z_prev[uv_others[self.N_surv_samples + q]]])
                    embeddings_non2.extend([z_prev[uv_others[q]], z_prev[v_it]])
                else:
                    for k_ in range(2):
                        lambda_uv_t_non_events.append(
                            self.intensity_rate_lambda(z_prev[u_it],
                                                       z_prev[uv_others[q]], torch.zeros(1).long() + k_))
                        lambda_uv_t_non_events.append(
                            self.intensity_rate_lambda(z_prev[uv_others[self.N_surv_samples + q]],
                                                       z_prev[v_it],
                                                       torch.zeros(1).long() + k_))


            # compute conditional density for all possible pairs
            # here it's important NOT to use any information that the event between nodes u,v has happened
            # so, we use node embeddings of the previous time step: z_prev
            if self.exp or not self.training:
                with torch.no_grad():
                    z_cat = torch.cat((z_prev[u_it].detach().unsqueeze(0).expand(N, -1),
                                       z_prev[v_it].detach().unsqueeze(0).expand(N, -1)), dim=0)
                    Lambda = self.intensity_rate_lambda(z_cat, z_prev.detach().repeat(2, 1),
                                                        torch.zeros(len(z_cat)).long() + k).detach()
                    if not self.training:
                        A_pred[it, u_it, :] = Lambda[:N]
                        A_pred[it, v_it, :] = Lambda[N:]

                        assert torch.sum(torch.isnan(A_pred[it])) == 0, (it, torch.sum(torch.isnan(A_pred[it])))
                        if self.exp:
                            # Compute the survival term (See page 3 in the paper)
                            # we only need to compute the term for rows u_it and v_it in our matrix s to save time
                            # because we will compute rank only for nodes u_it and v_it
                            s1 = self.cond_density(time_bar[it], time_cur[it], u_it, v_it)
                            Surv[it, [u_it, v_it], :] = s1

                    if self.exp:
                        time_key = int(time_cur[it].item())
                        idx = np.delete(np.arange(N), [u_it, v_it])  # nonevents for node u
                        idx = np.concatenate((idx, idx + N))   # concat with nonevents for node v

                        if len(self.time_keys) >= len(self.Lambda_dict):
                            # shift in time (remove the oldest record)
                            time_keys = np.array(self.time_keys)
                            time_keys[:-1] = time_keys[1:]
                            self.time_keys = list(time_keys[:-1])  # remove last
                            self.Lambda_dict[:-1] = self.Lambda_dict.clone()[1:]
                            self.Lambda_dict[-1] = 0

                        self.Lambda_dict[len(self.time_keys)] = Lambda[idx].sum().detach()  # total intensity of non events for the current time step
                        self.time_keys.append(time_key)

            # Once we made predictions for the training and test sample, we can update node embeddings
            z_all.append(z_new)
            # update S
            if self.generate or (not self.training and self.latent_graph and self.encoder is not None):
                S_tmp, logits_tmp, reg2 = self.generate_S(z_new, u_it, v_it, train_enc=self.training and self.train_enc)
                if self.training:
                    reg = reg + reg2

                self.S = self.S.clone()

                self.S[u_it, v_it] = S_tmp[0, 1]
                self.S[v_it, u_it] = S_tmp[1, 0]

                self.S = self.S / (torch.sum(self.S, dim=1, keepdim=True) + 1e-7)
                self.logits[u_it, v_it] = logits_tmp[0, 1]
                self.logits[v_it, u_it] = logits_tmp[1, 0]
                self.A = self.S
                S_batch.append(self.S.data.cpu().numpy())


            self.t_p += 1


        self.z = z_new  # update node embeddings

        # Batch update
        if self.opt:
            lambda_uv_t = self.intensity_rate_lambda(torch.stack(embeddings1, dim=0),
                                                     torch.stack(embeddings2, dim=0), event_types)
            non_events = len(embeddings_non1)
            n_types = 2
            lambda_uv_t_non_events = torch.zeros(non_events * n_types, device=self.device)
            embeddings_non1 = torch.stack(embeddings_non1, dim=0)
            embeddings_non2 = torch.stack(embeddings_non2, dim=0)
            idx = None
            empty_t = torch.zeros(non_events, dtype=torch.long)
            types_lst = torch.arange(n_types)
            for k in types_lst:
                if idx is None:
                    idx = np.arange(non_events)
                else:
                    idx += non_events
                lambda_uv_t_non_events[idx] = self.intensity_rate_lambda(embeddings_non1, embeddings_non2, empty_t + k)

            if update_attn:
                # update only once per batch
                for it, k in enumerate(event_types):
                    u_it, v_it = u_all[it], v_all[it]
                    self.update_S_A(u_it, v_it, k.item(), node_degrees[it], lambda_uv_t[it].item())


        else:
            lambda_uv_t = torch.cat(lambda_uv_t)
            lambda_uv_t_non_events = torch.cat(lambda_uv_t_non_events)

        if len(S_batch) > 0:
            S_batch = np.stack(S_batch)

        if len(reg) > 1:
            reg = [torch.stack(reg).mean()]

        return lambda_uv_t, lambda_uv_t_non_events / self.N_surv_samples, [A_pred, Surv], S_batch, reg
