#import matplotlib.pyplot as plt
import numpy as np


class FreqBaseline():

    def __init__(self, train, test):
        N = train.N_nodes

        H_train = np.zeros((N, N))
        n = 0
        for e in train.all_events:
            H_train[e[0], e[1]] += 1
            H_train[e[1], e[0]] += 1
            n += 1
        print('train', n, H_train.max(), H_train.min(), H_train.std())
        self.H_train = H_train

        self.H_train_norm = self.H_train / (np.sum(self.H_train, axis=1, keepdims=True) + 1e-10)

        H_test = np.zeros((N, N))
        c = 0
        for e in test.all_events:
            H_test[e[0], e[1]] += 1
            c += 1
        print('test', c, H_test.max(), H_test.min(), H_test.std())
        self.H_test = H_test

        print('\nFrequency baselines'.upper())

        R = np.corrcoef(H_train.flatten(), H_test.flatten())[0, 1]
        print('correlation coefficient between train and test events: %f' % R)

        # Frequency based baseline
        ranks = []
        hits_10 = []
        for b in range(len(test.all_events)):
            u_it, v_it, k, dt = test.all_events[b]
            assert u_it != v_it, ('loops are not permitted', u_it, v_it, k[b])
            idx1 = list(np.argsort(H_train[u_it])[::-1])
            idx1.remove(u_it)
            idx2 = list(np.argsort(H_train[v_it])[::-1])
            idx2.remove(v_it)
            rank, hits = self.get_mar_hits_sym(idx1, idx2, v_it, u_it)
            ranks.append(rank)
            hits_10.append(hits)
        print('\nFrequency MAR and HITS@10:', np.mean(ranks), np.std(ranks), np.mean(hits_10), np.std(hits_10))

        self.results = {}

        self.results['freq'] = (np.mean(ranks), np.mean(hits_10))

        A_all, keys, _ = train.get_Adjacency(multirelations=True)

        if len(A_all.shape) == 2:
            A_all = A_all[:, :, None]

        for rel_ind, rel in enumerate(keys):
            # print(A_all.shape, rel_ind, keys)
            A = A_all[:, :, rel_ind]
            ranks = []
            hits_10 = []
            for b in range(len(test.all_events)):
                u_it, v_it, k, dt = test.all_events[b]
                assert u_it != v_it, ('loops are not permitted', u_it, v_it, k[b])

                idx1 = list(np.where(A[u_it] == 1)[0])
                if u_it in idx1:
                    idx1.remove(u_it)

                idx2 = list(np.where(A[v_it] == 1)[0])
                if v_it in idx2:
                    idx2.remove(v_it)

                np.random.shuffle(idx1)
                np.random.shuffle(idx2)

                idx1 += list(self.permute_array(np.where(A[u_it] == 0)[0]))
                idx2 += list(self.permute_array(np.where(A[v_it] == 0)[0]))

                if u_it in idx1:
                    idx1.remove(u_it)
                if v_it in idx2:
                    idx2.remove(v_it)

                rank, hits = self.get_mar_hits_sym(idx1, idx2, v_it, u_it)
                ranks.append(rank)
                hits_10.append(hits)
            print('%s: Frequency MAR and HITS@10:' % rel, np.mean(ranks), np.std(ranks), np.mean(hits_10), np.std(hits_10))

            self.results[rel] = (np.mean(ranks), np.mean(hits_10))

        ranks = []
        hits_10 = []
        for b in range(len(test.all_events)):
            u_it, v_it, k, dt = test.all_events[b]
            assert u_it != v_it, ('loops are not permitted', u_it, v_it, k[b])

            idx1 = list(self.permute_array(np.arange(N)))
            if u_it in idx1:
                idx1.remove(u_it)

            idx2 = list(self.permute_array(np.arange(N)))
            if v_it in idx2:
                idx2.remove(v_it)

            rank, hits = self.get_mar_hits_sym(idx1, idx2, v_it, u_it)
            ranks.append(rank)
            hits_10.append(hits)

        print('Random: Frequency MAR and HITS@10:', np.mean(ranks), np.std(ranks), np.mean(hits_10), np.std(hits_10))
        self.results['random'] = (np.mean(ranks), np.mean(hits_10))

        for rel_ind, rel in enumerate(keys):
            A = A_all[:, :, rel_ind]
            ranks = []
            hits_10 = []
            for b in range(len(test.all_events)):
                u_it, v_it, k, dt = test.all_events[b]
                assert u_it != v_it, ('loops are not permitted', u_it, v_it, k[b])

                idx1 = list(np.where(A[u_it] == 1)[0])
                if u_it in idx1:
                    idx1.remove(u_it)

                idx2 = list(np.where(A[v_it] == 1)[0])
                if v_it in idx2:
                    idx2.remove(v_it)

                if len(idx1) > 0:
                    idx1 = np.array(idx1)
                    idx1 = list(idx1[np.argsort(H_train[u_it, idx1])[::-1]])
                if len(idx2) > 0:
                    idx2 = np.array(idx2)
                    idx2 = list(idx2[np.argsort(H_train[v_it, idx2])[::-1]])

                idx1 += list(self.permute_array(np.where(A[u_it] == 0)[0]))
                idx2 += list(self.permute_array(np.where(A[v_it] == 0)[0]))

                if u_it in idx1:
                    idx1.remove(u_it)
                if v_it in idx2:
                    idx2.remove(v_it)

                rank, hits = self.get_mar_hits_sym(idx1, idx2, v_it, u_it)
                ranks.append(rank)
                hits_10.append(hits)
            print('%s: Frequency overlap MAR and HITS@10:' % rel, np.mean(ranks), np.std(ranks), np.mean(hits_10), np.std(hits_10))
            self.results['%s_overlap' % rel] = (np.mean(ranks), np.mean(hits_10))

        print('\n')

    def get_mar_hits_sym(self, idx1, idx2, v, u, k=10):
        rank1, hits1 = self.get_mar_hits(idx1, v, k=k)  # get nodes most likely connected to u[b] and find out the rank of v[b] among those nodes
        rank2, hits2 = self.get_mar_hits(idx2, u, k=k)  # get nodes most likely connected to v[b] and find out the rank of u[b] among those nodes
        return (rank1 + rank2) / 2, (hits1 + hits2) / 2

    def get_mar_hits(self, idx, v, k=10):
        rank = np.where(np.array(idx) == v)[0]
        assert len(rank) == 1, (rank, idx, v)
        return rank[0], float(rank[0] < k)

    def permute_array(self, arr):
        return arr[np.random.permutation(len(arr))]
