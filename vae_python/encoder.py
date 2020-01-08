import tensorflow as tf


class Encoder(tf.keras.Model):
    def __init__(self, hidden_size, vocab_size, batch_size=1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.fw_layer = tf.keras.layers.LSTM(self.hidden_size,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.bw_layer = tf.keras.layers.LSTM(self.hidden_size,
                                       return_sequences=True,
                                       return_state=True,
                                       go_backwards=True,
                                       recurrent_initializer='glorot_uniform')
        self.bi_rnn = tf.keras.layers.Bidirectional(self.fw_layer, merge_mode='concat', backward_layer=self.bw_layer)
    def call(self, d):
        output,fw_hidden,fw_c,bw_hidden,bw_c = self.bi_rnn(d)
        hidden = tf.concat([fw_hidden,bw_hidden],1)
        c = tf.concat([fw_c,bw_c],1)
        return output, hidden,c