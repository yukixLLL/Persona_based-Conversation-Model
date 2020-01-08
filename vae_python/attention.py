import tensorflow as tf

class Attention_Feed(tf.keras.Model):
    def __init__(self, hidden_size):
        super(Attention_Feed, self).__init__()
        self.W1 = tf.keras.layers.Dense(hidden_size)
        self.W2 = tf.keras.layers.Dense(hidden_size)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        hidden_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights