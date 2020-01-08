from attention import *
from param import DROP_OUT, speakerNum

class Decoder(tf.keras.Model):
    def __init__(self, hidden_size, vocab_size, embedding_dim,speaker_dim, num_layers=1):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.speaker_embedding = tf.keras.layers.Embedding(speakerNum,speaker_dim)
        self.input_size = embedding_dim
        self.output_size = vocab_size  # vocabulary size
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
        self.fc = tf.keras.layers.Dense(self.output_size)
        self.W1 = tf.keras.layers.Dense(speaker_dim)
        self.W2 = tf.keras.layers.Dense(speaker_dim)
        # attention feed on context
        self.attention = Attention_Feed(self.hidden_size)

    def call(self, x, enc_output, init_state, speaker_id, addressee_id=None):
        #         batch_size = x.size()[1]
        hidden = init_state[0]
        context_vector, attention_weights = self.attention(hidden, enc_output)
        features = self.embedding(x)
        # personas
        speaker = self.speaker_embedding(speaker_id)
        if addressee_id is not None:
            addressee = self.speaker_embedding(addressee_id)
            v_ij = self.combine_user_vector(speaker, addressee)
            features = tf.concat([features, tf.expand_dims(v_ij, 1)], axis=-1)
        else:
            features = tf.concat([features, tf.expand_dims(speaker, 1)], axis=-1)
        #         max_length = enc_output.size(0)
        r = tf.concat([tf.expand_dims(context_vector, 1), features], axis=-1)

        # passing the concatenated vector to the 4-layer LSTM
        output, hidden, c = self.lstm_1(r, initial_state=init_state)
        init_state = [hidden, c]
        for k in range(self.num_layers - 1):
            output, state, c = self.lstms[k](output, initial_state=init_state)
            init_state = [hidden, c]

        # Removes dimensions of size 1 from the shape of a tensor.
        # output shape: (batch_size, 1, hidden_size) --> (batch_size *1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # before fc: output shape == （batch_size, hidden_size)
        # after fc: output shape == （batch_size, vocab_size)
        output = tf.nn.log_softmax(self.fc(output), axis=1)
        return output, state, c, attention_weights

    def combine_user_vector(self,i_em, j_em):
        # size = i_em.shape[-1]
        V_ij = tf.nn.tanh(self.W1(i_em) + self.W2(j_em))
        return V_ij