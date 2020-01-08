import tensorflow as tf

class Word_Embedding(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim):
        super(Word_Embedding, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

    def call(self, x):
        return self.embedding(x)