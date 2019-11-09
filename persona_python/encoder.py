import tensorflow as tf
from param import DROP_OUT

class Encoder(tf.keras.Model):
    def __init__(self, hidden_size, vocab_size,embedding_dim, num_layers=1, batch_size=1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.input_size = embedding_dim
        self.lstm_1 = tf.keras.layers.LSTM(self.hidden_size,
                                           return_sequences=True,
                                           return_state=True,
                                           dropout=DROP_OUT,
                                           recurrent_initializer='glorot_uniform')
        self.lstms = []
        for k in range(self.num_layers - 1):
            self.lstms.append(tf.keras.layers.LSTM(self.hidden_size,
                                                   return_sequences=True,
                                                   return_state=True,
                                                   dropout=DROP_OUT))
    def call(self, d, init_state):
        d = self.embedding(d)
        output, hidden, c = self.lstm_1(d, initial_state = init_state)
        init_state = [hidden, c]
        # four layer train, 4 lstm
        for k in range(self.num_layers - 1):
            output, hidden,c = self.lstms[k](output, initial_state = init_state)
            init_state = [hidden, c]
        return output, hidden,c

    def initialize_hidden_state(self,batch_size=0):
        if batch_size == 0: batch_size =self.batch_size
        init_hidden = tf.zeros((batch_size, self.hidden_size))
        init_c = tf.zeros((batch_size, self.hidden_size))
        return [init_hidden,init_c]