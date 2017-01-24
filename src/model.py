import numpy as np
import pandas as pd
import tensorflow as tf

class Model:
    def __init__(self, num_feat, lstm_size, fc_hidd_size, lr):
        '''
        Initializes the LSTM model
        
        Inputs:
        - num_feat: An integer denoting the number of features (number of features is fixed across all instruments and timestamps)
        - lstm_size: An integer defining the number of hidden units in the LSTM layer
        - fc_hidd_size: An integer representing the size of the hidden fully connected layer
        - lr: An integer for initializing the ADAM optimizer
        
        Returns:
        - The initialized LSTM model
        '''
        
        self.num_feat = num_feat
        self.lstm_size = lstm_size
        self.fc_hidd_size = fc_hidd_size
        self.lr = lr
        self.build_graph()
        
    def build_graph(self):
        '''Builds the TF graph'''
        
        '''        
        Inputs:
        - X: A placeholder for a tensor of shape [batch_size, time_stamp, num_features]. Note, the only fixed dimension is the num_feat. batch_size depends on the number of observed instruments at each time_stamp. time_stamp depends on the length of the observations for different instruments. X contains all the observations for a batch of instruments.
        - Y: A placeholder for a tensor of shape [batch_size, time_stamp, 1]. The third dimension is a dummy dimension to match the shape of the labels with the prediction tensor at the end. Y contains the target values for each instrument at each timestamp.
        - seqeunce_length: A placeholder for a tensor of shape [batch_size]. sequence_length denotes the number of observation for each instrument.
        - target_mask: A placeholder for a tensor with the same shape as "Y". This tensor only contains "0" or "1". The mask is used to kill the gradient flowing back in the back-propagation for the {instrument, timestamp} pairs that are not included in the observations.
        '''
        self.X = tf.placeholder(dtype=tf.float64, shape=[None, None, self.num_feat], name='X')
        self.y = tf.placeholder(dtype=tf.float64, shape=[None, None, 1], name='y')
        self.sequence_length = tf.placeholder(dtype=tf.int64, shape=[None], name='sequence_length')
        self.target_mask = tf.placeholder(dtype=tf.float64, shape=[None, None, 1], name='target_mask')
        
        '''
        Model:
        - state_c, state_h: Are tensors of shape [batch_size, 4*lstm_size]
        - dynamic_rnn: Takes a state tuple to initialize the lstm_cell. We use the state tuple to initialize the lstm layer to the last observation. Maintaining states for each instrument happens out side the model.
        - pred_mask: performs a elementwise multiplication with target_mask to prevent the gradient from flowing back.
        '''
        lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.lstm_size, initializer=tf.contrib.layers.xavier_initializer())
        # The initial state of the LSTM
        state_shape = [None, self.lstm_size]
        self.state_c = tf.placeholder(dtype=tf.float64, shape=state_shape)
        self.state_h = tf.placeholder(dtype=tf.float64, shape=state_shape)
        initial_state = tf.nn.rnn_cell.LSTMStateTuple(self.state_c, self.state_h)
        
        self.output, self.state = tf.nn.dynamic_rnn(
            cell=lstm_cell,
            dtype=tf.float64,
            sequence_length=self.sequence_length,
            initial_state=initial_state,
            inputs=self.X
        )
        
        self.f0 = tf.contrib.layers.fully_connected(
            self.output, num_outputs=self.fc_hidd_size,
            activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
        self.pred = tf.contrib.layers.fully_connected(
            self.f0, num_outputs=1, activation_fn=None,
            weights_initializer=tf.contrib.layers.xavier_initializer())
        self.pred_masked = tf.mul(self.pred, self.target_mask)
        
        '''Loss'''
        self.errors = tf.squared_difference(self.y, self.pred_masked)
        self.loss = tf.reduce_mean(self.errors)
        
        '''Optimize'''
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.lr,
            beta1=0.9, beta2=0.999,
            epsilon=1e-08,
            use_locking=False,
            name='Adam'
        ).minimize(self.loss)
        
    def train_on_batch(self, session, sequence, sequence_targets, target_mask, sequence_length):
        '''
        Inputs:
        - session: A TF session
        - sequence: A numpy array of shape [batch_size, time_stamp, num_feat]
        - sequence_targets: A numpy array of shape [batch_size, time_stamp, 1]
        - target_mask: A numpy array like the sequence_targets
        - sequence_length: A numpy array of shape [batch_size]
        
        Return:
        - loss: mean squared error
        
        Note:
        During training we pass a complete sequence to the model, therefore, the lstm state is initialized to all zeros.
        '''
        batch_size = len(sequence)
        
        feed_dict = {
            self.X: sequence,
            self.y: sequence_targets,
            self.target_mask: target_mask,
            self.sequence_length: sequence_length,
            self.state_c: np.zeros(shape=[batch_size, self.lstm_size]),
            self.state_h: np.zeros(shape=[batch_size, self.lstm_size])
        }

        loss, _ = session.run([self.loss, self.optimizer], feed_dict=feed_dict)
        return loss
    
    def predict(self, session, sequence, sequence_length, initial_state=None):
        '''
        Inputs:
        - session: A TF session
        - sequence: A numpy array of shape [batch_size, time_stamp, num_feat]
        - sequence_length: A numpy array of shape [batch_size]
        - initial_state: A numpy array of shape [batch_size, 4*lstm_size], containing the lstm_cell state for each instrument after the last observation. The default value is all zeros.
        
        Return:
        - pred: Predicted "y" values
        - state: Final state of the lstm_cell
        '''
        feed_dict = {self.X: sequence, self.sequence_length: sequence_length}
        if initial_state is None:
            batch_size = len(sequence)
            feed_dict[self.state_c] = np.zeros(shape=[batch_size, self.lstm_size])
            feed_dict[self.state_h] = np.zeros(shape=[batch_size, self.lstm_size])
        else:
            feed_dict[self.state_c] = initial_state.c
            feed_dict[self.state_h] = initial_state.h
        
        pred, state = session.run([self.pred, self.state], feed_dict=feed_dict)
        return pred, state        
        
    def fit(self, session, input_df, num_epoch, batch_size):
        '''
        This function is specific to the two_sigma competition. It receives a dataframe and prepares all the required inputs (sequence, sequence_targets, target_mask, sequence_length) for training the model. At each iteration, it selects a set of random instruments of size batch_size and calls the train_on_batch function.
        
        Inputs:
        - session: A TF session
        - input_df: A pandas dataframe containing the features, 'id', 'y', and 'timestamp' columns
        - num_epoch: An integer denoting the number passes over all the training data
        - batch_szie: An integer denoting the number of streams to be passed at each iteration
        '''
        examples = {}
        for _id, df in input_df.groupby('id'):
            exp = []
            exp.append(df.drop(['id', 'y'], axis=1).values)
            exp.append(df['y'].values)
            exp.append(df.shape[0])
            examples[_id] = exp
                
        keys = examples.keys()
        num_seq = len(keys)
        num_batch = ((num_seq / batch_size) + 1) * num_epoch
        
        for _ in range(num_batch):
            batch_keys = np.random.choice(keys, batch_size, False)
            batch = [examples[k] for k in batch_keys]
            X, y, ln = zip(*batch)
            
            max_len = max(ln)
            feat = [np.pad(s, ((0, max_len - s.shape[0]), (0, 0)), 'constant') for s in X]
            tar = [np.pad(t, ((0, max_len - t.shape[0])), 'constant') for t in y]
            mask = [np.pad(np.ones_like(t), ((0, max_len - t.shape[0])), 'constant') for t in y]
            feat, tar, mask = np.array(feat), np.expand_dims(np.array(tar), 2), np.expand_dims(np.array(mask), 2)
            
            loss = self.train_on_batch(session, feat, tar, mask, ln)
            yield loss[0]