import os
import numpy as np
import datetime
import pickle
from datetime import datetime, timezone
import dateutil.parser
from data_loader import EventsDataset


def iso_parse(dt):
    # return datetime.fromisoformat(dt)  # python >= 3.7
    return dateutil.parser.isoparse(dt)

class GithubDataset(EventsDataset):

    def __init__(self, split, data_dir='./Github'):
        super(GithubDataset, self).__init__()

        if split == 'train':
            time_start = 0
            time_end = datetime(2013, 8, 31, tzinfo=self.TZ).toordinal()
        elif split == 'test':
            time_start = datetime(2013, 9, 1, tzinfo=self.TZ).toordinal()
            time_end = datetime(2014, 1, 1, tzinfo=self.TZ).toordinal()
        else:
            raise ValueError('invalid split', split)

        self.FIRST_DATE = datetime(2012, 12, 28, tzinfo=self.TZ)

        self.TEST_TIMESLOTS = [datetime(2013, 9, 1, tzinfo=self.TZ),
                               datetime(2013, 9, 25, tzinfo=self.TZ),
                               datetime(2013, 10, 20, tzinfo=self.TZ),
                               datetime(2013, 11, 15, tzinfo=self.TZ),
                               datetime(2013, 12, 10, tzinfo=self.TZ),
                               datetime(2014, 1, 1, tzinfo=self.TZ)]



        with open(os.path.join(data_dir, 'github_284users_events_2013.pkl'), 'rb') as f:
            users_events, event_types = pickle.load(f)

        with open(os.path.join(data_dir, 'github_284users_follow_2011_2012.pkl'), 'rb') as f:
            users_follow = pickle.load(f)

        print(event_types)

        self.events2name = {}
        for e in event_types:
            self.events2name[event_types[e]] = e
        print(self.events2name)

        self.event_types = ['ForkEvent', 'PushEvent', 'WatchEvent', 'IssuesEvent', 'IssueCommentEvent',
                           'PullRequestEvent', 'CommitCommentEvent']
        self.assoc_types = ['FollowEvent']
        self.is_comm = lambda d: self.events2name[d['type']] in self.event_types
        self.is_assoc = lambda d: self.events2name[d['type']] in self.assoc_types

        user_ids = {}
        for id, user in enumerate(sorted(users_events)):
            user_ids[user] = id

        self.N_nodes = len(user_ids)

        self.A_initial = np.zeros((self.N_nodes, self.N_nodes))
        for user in users_follow:
            for e in users_follow[user]:
                assert e['type'] in self.assoc_types, e['type']
                if e['login'] in users_events:
                    self.A_initial[user_ids[user], user_ids[e['login']]] = 1

        self.A_last = np.zeros((self.N_nodes, self.N_nodes))
        for user in users_events:
            for e in users_events[user]:
                if self.events2name[e['type']] in self.assoc_types:
                    self.A_last[user_ids[user], user_ids[e['login']]] = 1

        print('\nA_initial', np.sum(self.A_initial))
        print('A_last', np.sum(self.A_last), '\n')

        all_events = []
        for user in users_events:
            if user not in user_ids:
                continue
            user_id = user_ids[user]
            for ind, event in enumerate(users_events[user]):
                event['created_at'] = datetime.fromtimestamp(event['created_at'])
                if event['created_at'].toordinal() >= time_start and event['created_at'].toordinal() <= time_end:
                    if 'owner' in event:
                        if event['owner'] not in user_ids:
                            continue
                        user_id2 = user_ids[event['owner']]
                    elif 'login' in event:
                        if event['login'] not in user_ids:
                            continue
                        user_id2 = user_ids[event['login']]
                    else:
                        raise ValueError('invalid event', event)
                    if user_id != user_id2:
                        all_events.append((user_id, user_id2,
                                           self.events2name[event['type']], event['created_at']))

        self.all_events = sorted(all_events, key=lambda t: t[3].timestamp())
        print('\n%s' % split.upper())
        print('%d events between %d users loaded' % (len(self.all_events), self.N_nodes))
        print('%d communication events' % (len([t for t in self.all_events if t[2] == 1])))
        print('%d assocition events' % (len([t for t in self.all_events if t[2] == 0])))

        self.event_types_num = {self.assoc_types[0]: 0}
        k = 1  # k >= 1 for communication events
        for t in self.event_types:
            self.event_types_num[t] = k
            k += 1

        self.n_events = len(self.all_events)


    def get_Adjacency(self, multirelations=False):
        if multirelations:
            print('warning: Github has only one relation type (FollowEvent), so multirelations are ignored')
        return self.A_initial, self.assoc_types, self.A_last
