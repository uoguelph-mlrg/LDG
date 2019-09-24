import matplotlib.pyplot as plt
import os
import sys
from os.path import join as pjoin
import numpy as np
import time
import datetime
import pickle
import copy
import pandas
import itertools
import torch
import torch.utils


def load_social_evolution_data(data_dir, prob, EVENT_TYPES, dump=True):
    data_file = pjoin(data_dir, 'data_prob%s.pkl' % prob)
    if os.path.isfile(data_file):
        print('loading data from %s' % data_file)
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
    else:
        data = {'initial_embeddings': SubjectsReader(pjoin(data_dir, 'Subjects.csv')).features_onehot}
        for split in ['train', 'test']:
            data.update({split: SocialEvolution(data_dir, split=split, MIN_EVENT_PROB=prob, EVENT_TYPES=EVENT_TYPES)})
        if dump:
            # dump data files to avoid their generation again
            print('saving data to %s' % data_file)
            with open(data_file, 'wb') as f:
                pickle.dump(data, f, protocol=2)  # for compatibility
    return data


class SocialEvolutionTorch(torch.utils.data.Dataset):
    '''
    Class to load batches for training and testing
    '''

    def __init__(self,
                 subj_features,
                 data,
                 event_types,
                 FIRST_DATE,
                 MainAssociation,
                 data_train=None):
        self.subj_features = subj_features
        self.data = data
        self.event_types = event_types
        self.all_events = []
        self.event_types_num = {}
        self.time_bar = None
        self.FIRST_DATE = FIRST_DATE
        self.MainAssociation = MainAssociation

        k = 1  # k >= 1 for communication events
        for t in event_types:
            print(t, k, len(data.EVENT_TYPES[t].tuples))

            events = list(filter(lambda x: x[3].toordinal() >= FIRST_DATE.toordinal(), self.data.EVENT_TYPES[t].tuples))
            self.all_events.extend(events)
            self.event_types_num[t] = k
            k += 1

        n = len(self.all_events)

        # Initial topology
        if len(list(data.Adj.keys())) > 0:

            keys = sorted(list(data.Adj[list(data.Adj.keys())[0]].keys()))  # relation keys
            keys.remove(MainAssociation)
            keys = [MainAssociation] + keys  # to make sure CloseFriend goes first

            k = 0  # k <= 0 for association events
            for rel in keys:

                if rel != MainAssociation:
                    continue
                if data_train is None:
                    date = sorted(list(data.Adj.keys()))[0]  # first date
                    Adj_prev = data.Adj[date][rel]
                else:
                    date = sorted(list(data_train.Adj.keys()))[-1]  # last date of the training set
                    Adj_prev = data_train.Adj[date][rel]
                self.event_types_num[rel] = k

                N = Adj_prev.shape[0]

                # Associative events
                for date_id, date in enumerate(sorted(list(data.Adj.keys()))):  # start from the second survey
                    if date.toordinal() >= FIRST_DATE.toordinal():
                        # for rel_id, rel in enumerate(sorted(list(dygraphs.Adj[date].keys()))):
                        assert data.Adj[date][rel].shape[0] == N
                        for u in range(N):
                            for v in range(u + 1, N):
                                # if two nodes become friends, add the event
                                if data.Adj[date][rel][u, v] > 0 and Adj_prev[u, v] == 0:
                                    assert u != v, (u, v, k)
                                    self.all_events.append((u, v, rel, date))

                    Adj_prev = data.Adj[date][rel]

                k -= 1

                print(data.split, rel, len(self.all_events) - n)
                n = len(self.all_events)

        self.all_events = sorted(self.all_events, key=lambda x: int(x[3].timestamp()))

        print('%d events' % len(self.all_events))
        print('last 10 events:')
        for event in self.all_events[-10:]:
            print(event)

        self.n_samples = len(self.all_events)

        H_train = np.zeros((N, N))
        c = 0
        for e in self.all_events:
            H_train[e[0], e[1]] += 1
            H_train[e[1], e[0]] += 1
            c += 1
        print('H_train', c, H_train.max(), H_train.min(), H_train.std())
        self.H_train = H_train


    def get_Adjacency(self, multirelations=False):
        Adj_all = self.data.Adj[sorted(list(self.data.Adj.keys()))[0]]
        Adj_all_last = self.data.Adj[sorted(list(self.data.Adj.keys()))[-1]]
        Adj_friends = Adj_all[self.MainAssociation].copy()
        if multirelations:
            keys = sorted(list(Adj_all.keys()))
            keys.remove(self.MainAssociation)
            keys = [self.MainAssociation] + keys  # to make sure CloseFriend goes first
            Adj_all = np.stack([Adj_all[rel].copy() for rel in keys], axis=2)
            Adj_all_last = np.stack([Adj_all_last[rel].copy() for rel in keys], axis=2)
        else:
            keys = [self.MainAssociation]
            Adj_all = Adj_all[self.MainAssociation].copy()
            Adj_all_last = Adj_all_last[self.MainAssociation].copy()

        return Adj_all, Adj_all_last, Adj_friends, keys


    def time_to_onehot(self, d):
        x = []
        for t, max_t in [(d.weekday(), 7), (d.hour, 24), (d.minute, 60), (d.second, 60)]:
            x_t = np.zeros(max_t)
            x_t[t] = 1
            x.append(x_t)
        return np.concatenate(x)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):

        tpl = self.all_events[index]
        u, v, rel, time_cur = tpl

        # Compute time delta in seconds (t_p - \bar{t}_p_j) that will be fed to W_t
        time_delta_uv = np.zeros((2, 4))  # two nodes x 4 values

        # most recent previous time for all nodes
        time_bar = self.time_bar.copy()

        assert u != v, (tpl, self.event_types_num[rel])

        for c, j in enumerate([u, v]):
            # if node_global_time[j] == 0 then time delta is 0 - no previous time stamps

            t = datetime.datetime.fromtimestamp(self.time_bar[j])
            if t.toordinal() >= self.FIRST_DATE.toordinal():  # assume no events before FIRST_DATE
                td = time_cur - t
                time_delta_uv[c] = np.array([td.days,  # total number of days, still can be a big number
                                             td.seconds // 3600,  # hours, max 24
                                             (td.seconds // 60) % 60,  # minutes, max 60
                                             td.seconds % 60],  # seconds, max 60
                                             np.float)
                # assert time_delta_uv.min() >= 0, (index, tpl, time_delta_uv[c], node_global_time[j])
            else:
                raise ValueError('unexpected result', t, self.FIRST_DATE)
            self.time_bar[j] = time_cur.timestamp()  # last time stamp for nodes u and v

        k = self.event_types_num[rel]

        assert k in np.arange(0, 4), k

        assert np.float64(time_cur.timestamp()) == time_cur.timestamp(), (np.float64(time_cur.timestamp()), time_cur.timestamp())
        time_cur = np.float64(time_cur.timestamp())
        time_bar = time_bar.astype(np.float64)
        time_cur = torch.from_numpy(np.array([time_cur])).double()
        assert time_bar.max() <= time_cur, (time_bar.max(), time_cur)
        return u, v, time_delta_uv, k, time_bar, time_cur

class CSVReader:
    '''
    General class to read any relationship csv in this dataset
    '''

    def __init__(self,
                 csv_path,
                 split,  # 'train', 'test', 'all'
                 MIN_EVENT_PROB,
                 event_type=None,
                 N_subjects=None,
                 test_slot=1):
        self.csv_path = csv_path
        print(os.path.basename(csv_path))

        if split == 'train':
            time_start = 0
            time_end = datetime.datetime(2009, 4, 30).toordinal()
        elif split == 'test':
            if test_slot != 1:
                raise NotImplementedError('test on time slot 1 for now')
            time_start = datetime.datetime(2009, 5, 1).toordinal()
            time_end = datetime.datetime(2009, 6, 30).toordinal()
        else:
            time_start = 0
            time_end = np.Inf

        csv = pandas.read_csv(csv_path)
        self.data = {}
        to_date1 = lambda s: datetime.datetime.strptime(s, '%Y-%m-%d')
        to_date2 = lambda s: datetime.datetime.strptime(s, '%Y-%m-%d %H:%M:%S')
        user_columns = list(filter(lambda c: c.find('user') >= 0 or c.find('id') >= 0, list(csv.keys())))
        assert len(user_columns) == 2, (list(csv.keys()), user_columns)
        self.time_column = list(filter(lambda c: c.find('time') >= 0 or c.find('date') >= 0, list(csv.keys())))
        assert len(self.time_column) == 1, (list(csv.keys()), self.time_column)
        self.time_column = self.time_column[0]

        self.prob_column = list(filter(lambda c: c.find('prob') >= 0, list(csv.keys())))

        for column in list(csv.keys()):
            values = csv[column].tolist()
            for fn in [int, float, to_date1, to_date2]:
                try:
                    values = list(map(fn, values))
                    break
                except Exception as e:
                    continue
            self.data[column] = values

        n_rows = len(self.data[self.time_column])

        time_stamp_days = np.array([d.toordinal() for d in self.data[self.time_column]], dtype=np.int)

        # skip data where one of users is missing (nan) or interacting with itself or timestamp not in range
        conditions = [~np.isnan(self.data[user_columns[0]]),
                      ~np.isnan(self.data[user_columns[1]]),
                      np.array(self.data[user_columns[0]]) != np.array(self.data[user_columns[1]]),
                      time_stamp_days >= time_start,
                      time_stamp_days <= time_end]

        if len(self.prob_column) == 1:
            print(split, event_type, self.prob_column)
            # skip data if the probability of event is 0 or nan (available for some event types)
            conditions.append(np.nan_to_num(np.array(self.data[self.prob_column[0]])) > MIN_EVENT_PROB)

        valid_ids = np.ones(n_rows, dtype=np.bool)
        for cond in conditions:
            valid_ids = valid_ids & cond

        self.valid_ids = np.where(valid_ids)[0]

        time_stamps_sec = [self.data[self.time_column][i].timestamp() for i in self.valid_ids]
        self.valid_ids = self.valid_ids[np.argsort(time_stamps_sec)]

        print(split, len(self.valid_ids), n_rows)

        for column in list(csv.keys()):
            values = csv[column].tolist()
            key = column + '_unique'
            for fn in [int, float, to_date1, to_date2]:
                try:
                    values = list(map(fn, values))
                    break
                except Exception as e:
                    continue

            self.data[column] = values

            values_valid = [values[i] for i in self.valid_ids]
            self.data[key] = np.unique(values_valid)
            print(key, type(values[0]), len(self.data[key]), self.data[key])

        self.subjects, self.time_stamps = [], []
        for usr_col in range(len(user_columns)):
            self.subjects.extend([self.data[user_columns[usr_col]][i] for i in self.valid_ids])
            self.time_stamps.extend([self.data[self.time_column][i] for i in self.valid_ids])

        # set O={(u, v, k, t)}
        self.tuples = []
        if N_subjects is not None:
            # Compute frequency of communcation between users
            print('user_columns', user_columns)
            self.Adj = np.zeros((N_subjects, N_subjects))
            for row in self.valid_ids:
                subj1 = self.data[user_columns[0]][row]
                subj2 = self.data[user_columns[1]][row]

                assert subj1 != subj2, (subj1, subj2)
                assert subj1 > 0 and subj2 > 0, (subj1, subj2)
                try:
                    self.Adj[int(subj1) - 1, int(subj2) - 1] += 1
                    self.Adj[int(subj2) - 1, int(subj1) - 1] += 1
                except:
                    print(subj1, subj2)
                    raise

                self.tuples.append((int(subj1) - 1,
                                    int(subj2) - 1,
                                    event_type,
                                    self.data[self.time_column][row]))

        n1 = len(self.tuples)
        self.tuples = list(set(itertools.chain(self.tuples)))
        self.tuples = sorted(self.tuples, key=lambda t: t[3].timestamp())
        n2 = len(self.tuples)
        print('%d/%d duplicates removed' % (n1 - n2, n1))


class SubjectsReader:
    '''
    Class to read Subjects.csv in this dataset
    '''

    def __init__(self,
                 csv_path):
        self.csv_path = csv_path
        print(os.path.basename(csv_path))

        csv = pandas.read_csv(csv_path)
        subjects = csv[list(filter(lambda column: column.find('user') >= 0, list(csv.keys())))[0]].tolist()
        print('Number of subjects', len(subjects))
        features = []
        for column in list(csv.keys()):
            if column.find('user') >= 0:
                continue
            values = list(map(str, csv[column].tolist()))
            features_unique = np.unique(values)
            features_onehot = np.zeros((len(subjects), len(features_unique)))
            for subj, feat in enumerate(values):
                ind = np.where(features_unique == feat)[0]
                assert len(ind) == 1, (ind, features_unique, feat, type(feat))
                features_onehot[subj, ind[0]] = 1
            features.append(features_onehot)

        features_onehot = np.concatenate(features, axis=1)
        print('features', features_onehot.shape)
        self.features_onehot = features_onehot


class SocialEvolution():
    '''
    Class to read all csv in this dataset
    '''

    def __init__(self,
                 data_dir,
                 split,
                 MIN_EVENT_PROB,
                 EVENT_TYPES=['SMS', 'Proximity', 'Calls']):
        self.data_dir = data_dir
        self.split = split
        self.MIN_EVENT_PROB = MIN_EVENT_PROB

        self.FIRST_DATE = datetime.datetime(2008, 9, 11)  # consider events starting from this time as in the DyRep paper
        self.relations = CSVReader(pjoin(data_dir, 'RelationshipsFromSurveys.csv'), split=split, MIN_EVENT_PROB=MIN_EVENT_PROB)
        self.relations.subject_ids = np.unique(self.relations.data['id.A'] + self.relations.data['id.B'])
        self.N_subjects = len(self.relations.subject_ids)
        print('Number of subjects', self.N_subjects)

        # Read communicative events
        self.EVENT_TYPES = {}
        for t in EVENT_TYPES:
            self.EVENT_TYPES[t] = CSVReader(pjoin(data_dir, '%s.csv' % t),
                                           split=split,
                                           MIN_EVENT_PROB=MIN_EVENT_PROB,
                                           event_type=t,
                                           N_subjects=self.N_subjects)

        # Compute adjacency matrices for associative relationship data
        self.Adj = {}
        dates = self.relations.data['survey.date']
        rels = self.relations.data['relationship']
        for date_id, date in enumerate(self.relations.data['survey.date_unique']):
            self.Adj[date] = {}
            ind = np.where(np.array([d == date for d in dates]))[0]
            for rel_id, rel in enumerate(self.relations.data['relationship_unique']):
                ind_rel = np.where(np.array([r == rel for r in [rels[i] for i in ind]]))[0]
                A = np.zeros((self.N_subjects, self.N_subjects))
                for j in ind_rel:
                    row = ind[j]
                    A[self.relations.data['id.A'][row] - 1, self.relations.data['id.B'][row] - 1] = 1
                    A[self.relations.data['id.B'][row] - 1, self.relations.data['id.A'][row] - 1] = 1
                self.Adj[date][rel] = A
                # sanity check
                for row in range(len(dates)):
                    if rels[row] == rel and dates[row] == date:
                        assert self.Adj[dates[row]][rels[row]][
                                   self.relations.data['id.A'][row] - 1, self.relations.data['id.B'][row] - 1] == 1
                        assert self.Adj[dates[row]][rels[row]][
                                   self.relations.data['id.B'][row] - 1, self.relations.data['id.A'][row] - 1] == 1
