import tensorflow as tf


class Prior_Network(tf.keras.Model):
    def __init__(self, latent_size):
        super(Prior_Network, self).__init__()
        self.fc = tf.keras.layers.Dense(max(latent_size * 2, 100))
        self.pri_nn = tf.keras.layers.Dense(latent_size * 2)

    def call(self, enc_hidden):
        prior_fc1 = self.fc(enc_hidden)
        prior_mulogvar = self.pri_nn(prior_fc1)
        prior_mu, prior_logvar = tf.split(prior_mulogvar, 2, axis=1)
        return prior_mu, prior_logvar
