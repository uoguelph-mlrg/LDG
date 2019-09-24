import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable


def MAR(A_pred, u, v, k, Survival_term):
    '''Computes mean average ranking for a batch of events'''
    ranks = []
    hits_10 = []
    N = len(A_pred)
    Survival_term = torch.exp(-Survival_term)
    A_pred *= Survival_term
    assert torch.sum(torch.isnan(A_pred)) == 0, (torch.sum(torch.isnan(A_pred)), Survival_term.min(), Survival_term.max())

    A_pred = A_pred.data.cpu().numpy()

    assert N == len(u) == len(v) == len(k), (N, len(u), len(v), len(k))
    for b in range(N):
        u_it, v_it = u[b].item(), v[b].item()
        assert u_it != v_it, (u_it, v_it, k[b])
        A = A_pred[b].squeeze()
        # remove same node
        idx1 = list(np.argsort(A[u_it])[::-1])
        idx1.remove(u_it)
        idx2 = list(np.argsort(A[v_it])[::-1])
        idx2.remove(v_it)
        rank1 = np.where(np.array(idx1) == v_it) # get nodes most likely connected to u[b] and find out the rank of v[b] among those nodes
        rank2 = np.where(np.array(idx2) == u_it)  # get nodes most likely connected to v[b] and find out the rank of u[b] among those nodes
        assert len(rank1) == len(rank2) == 1, (len(rank1), len(rank2))
        hits_10.append(np.mean([float(rank1[0] <= 9), float(rank2[0] <= 9)]))
        rank = np.mean([rank1[0], rank2[0]])
        assert isinstance(rank, np.float), (rank, rank1, rank2, u_it, v_it, idx1, idx2)
        ranks.append(rank)
    return ranks, hits_10


'''The functions below are copied from https://github.com/ethanfetaya/NRI'''

def my_softmax(input, axis=1):
    trans_input = input.transpose(axis, 0).contiguous()
    soft_max_1d = F.softmax(trans_input)
    return soft_max_1d.transpose(axis, 0)


def sample_gumbel(shape, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3
    Sample from Gumbel(0, 1)
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = torch.rand(shape).float()
    return - torch.log(eps - torch.log(U + eps))


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3
    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      tau: non-negative scalar temperature
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes
    Constraints:
    - this implementation only works on batch_size x num_features tensor for now
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = torch.zeros(*shape)
        if y_soft.is_cuda:
            y_hard = y_hard.cuda()
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y


def gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Draw a sample from the Gumbel-Softmax distribution

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda()
    y = logits + Variable(gumbel_noise)
    return my_softmax(y / tau, axis=-1)


def kl_categorical(preds, log_prior, num_atoms, eps=1e-16):
    kl_div = preds * (torch.log(preds + eps) - log_prior)
    return kl_div.sum() / (num_atoms * preds.size(0))


def kl_categorical_uniform(preds, num_atoms, num_edge_types, add_const=False,
                           eps=1e-16):
    kl_div = preds * torch.log(preds + eps)
    if add_const:
        const = np.log(num_edge_types)
        kl_div += const
    return kl_div.sum() / (num_atoms * preds.size(0))


def nll_gaussian(preds, target, variance, add_const=False):
    neg_log_p = ((preds - target) ** 2 / (2 * variance))
    if add_const:
        const = 0.5 * np.log(2 * np.pi * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0) * target.size(1))

