import tensorflow as tf

class Generation_Network(tf.keras.Model):
    def __init__(self, vocab_size, hidden_size):
        super(Generation_Network, self).__init__()
        self.bow_fc = tf.keras.layers.Dense(400)
        self.bow_logits = tf.keras.layers.Dense(vocab_size)
        self.init_fc = tf.keras.layers.Dense(hidden_size)

    def call(self, enc_hidden, latent_sample):
        gen_inputs = tf.concat([enc_hidden, latent_sample], 1)
        bow_fc1 = self.bow_fc(gen_inputs)
        bow_logit = self.bow_logits(bow_fc1)
        dec_init_state = self.init_fc(gen_inputs)
        # return decoder initial state and bag of word loss
        # we use lstm in decoder
        return [dec_init_state, dec_init_state], bow_logit