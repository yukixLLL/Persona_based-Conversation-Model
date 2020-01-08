import tensorflow as tf


class Recognition_Network(tf.keras.Model):
    def __init__(self, latent_size):
        super(Recognition_Network, self).__init__()
        self.rec_nn = tf.keras.layers.Dense(latent_size * 2)

    def call(self, enc_hidden, targ_hidden):
        recog_input = tf.concat([enc_hidden, targ_hidden], 1)
        recog_mulogvar = self.rec_nn(recog_input)
        recog_mu, recog_logvar = tf.split(recog_mulogvar, 2, axis=1)
        return recog_mu, recog_logvar