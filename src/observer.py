import numpy as np
from collections import defaultdict
from collections import namedtuple

class Observer:
    def __init__(self, lstm_size):
        self.state = {
            'c': defaultdict(lambda: np.zeros(lstm_size)),
            'h': defaultdict(lambda: np.zeros(lstm_size))
        }
        
    def update_state(self, ids, state):
        for i, _id in enumerate(ids):
            self.state['c'][_id] = state.c[i, :]
            self.state['h'][_id] = state.h[i, :]
            
    def last_state(self, ids):
        state = namedtuple('state', ['c', 'h'])
        state.c = np.vstack(map(lambda _id: self.state['c'][_id], ids))
        state.h = np.vstack(map(lambda _id: self.state['h'][_id], ids))
        return state
    
    def predict(self, df, M, sess):
        ids = df['id'].values
        state = self.last_state(ids)
        X = np.expand_dims(df.drop(['id', 'y'], axis=1).values, axis=1)
        l = np.ones(X.shape[0])
        
        pred, new_state = M.predict(session, X, l, state)
        self.update_state(ids, new_state)
        return pred