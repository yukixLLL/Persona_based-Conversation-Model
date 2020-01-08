import tensorflow as tf
from params import MAXLEN
def sample_gaussian(mu, logvar):
    epsilon = tf.random.normal(logvar.shape)
    std = tf.exp(0.5 * logvar)
    z= mu + tf.multiply(std, epsilon)
    return z

def count_real_word(real):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    mask = tf.cast(mask, dtype=tf.float32)
#     pdb.set_trace()
    word_per_line = tf.math.reduce_sum(mask,1)
    return word_per_line


def bow_loss_function(reals, bow_logit):
    # if I need to adjust according to the lenght of sentence???
    labels = reals[:, 1:]
    mask = tf.math.logical_not(tf.math.equal(labels, 0))

    bow_logit_tile = tf.tile(tf.expand_dims(bow_logit, 1), [1, MAXLEN - 1, 1])
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, bow_logit_tile)
    mask = tf.cast(mask, dtype=loss_.dtype)
    #     print("bow_loss.shape before reduction:{}".format(loss_.shape))
    bow_loss = tf.reduce_sum(loss_ * mask, axis=1)
    #     print("bow_loss.shape after reduction:{}".format(bow_loss.shape))
    return tf.reduce_mean(bow_loss)

def rc_loss_function(real, pred,word_per_line,validation=False):
    # loss of every word
    # true word mask
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ = loss_ * mask
    if validation:
        return loss_ / word_per_line
    else:
        return tf.math.reduce_sum(loss_/word_per_line)


def loss_function(avg_rc_loss, avg_bow_loss, recog_mu, recog_logvar, prior_mu, prior_logvar, global_iter, kl_full_step):
    temp = 1 + (recog_logvar - prior_logvar) - tf.math.divide(tf.pow(prior_mu - recog_mu, 2),
                                                              tf.exp(prior_logvar)) - tf.math.divide(tf.exp(recog_logvar), tf.exp(prior_logvar))
    kld = -1 / 2 * tf.reduce_sum(temp, axis=1)
    avg_kld = tf.reduce_mean(kld)
    kl_weight = tf.minimum(tf.cast(global_iter / kl_full_step, dtype=tf.float32), 1.0)

    elbo = avg_rc_loss + kl_weight * avg_kld
    aug_elbo = avg_bow_loss + elbo

    return elbo, aug_elbo, avg_kld