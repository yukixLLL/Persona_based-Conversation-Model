import tensorflow as tf
import numpy as np
import time
from util import *
from params import BATCH_SIZE

def validation(train_nn, inp, targ, speaker_id, addressee_id=None, batch_size=BATCH_SIZE):
    perplexity = 0
    elbo_losses = 0
    bow_losses = 0
    rc_losses = 0
    kl_losses = 0
    kl_fullstep = 1
    inp = np.asarray(inp)
    targ = np.asarray(targ)
    speaker_id = np.asarray(speaker_id)
    if addressee_id is not None:
        addressee_id = np.asarray(addressee_id)
    val_size = targ.shape[0]
    print("val_size: {}".format(val_size))
    num_batch = (int)(np.floor(val_size / batch_size))
    remaining_num = val_size - num_batch * batch_size
    Have_small_batch = True
    if remaining_num == 0:
        Have_small_batch = False
        remaining_num = batch_size
    for k in range(0, val_size, batch_size):
        start = time.time()
        size = batch_size
        if (k + batch_size) >= val_size:
            size = remaining_num
        inputs = inp[k:k + size]
        targs = targ[k:k + size]
        s_id = speaker_id[k:k + size]
        if addressee_id is not None:
            a_id = addressee_id[k:k + size]

        inp_emb = train_nn.word_embedding(inputs)
        targ_emb = train_nn.word_embedding(targs)
        enc_output, enc_hidden, enc_c = train_nn.encoder(inp_emb)
        _, targ_hidden, _ = train_nn.encoder(targ_emb)
        # VAE module
        prior_mu, prior_logvar = train_nn.prior_network(enc_hidden)
        recog_mu, recog_logvar = train_nn.recognition_network(enc_hidden, targ_hidden)

        latent_sample = tf.cond(False, lambda: sample_gaussian(prior_mu, prior_logvar),
                                lambda: sample_gaussian(recog_mu, recog_logvar))
        dec_init_state, bow_logit = train_nn.generation_network(enc_hidden, latent_sample)
        dec_input = tf.expand_dims(targ_emb[:, 0, :], 1)
        word_per_line = count_real_word(targs)
        rc_loss = 0
        rc_loss_sentence = 0
        for t in range(1, targs.shape[1]):
            if addressee_id is not None:
                predictions, dec_hidden, dec_c, _ = train_nn.decoder(dec_input, enc_output, dec_init_state, s_id, a_id)
            else:
                predictions, dec_hidden, dec_c, _ = train_nn.decoder(dec_input, enc_output, dec_init_state, s_id)

            dec_init_state = [dec_hidden, dec_c]
            loss_ = rc_loss_function(targs[:, t], predictions, word_per_line,True)
            rc_loss_sentence += loss_
            rc_loss += tf.math.reduce_sum(loss_)

            dec_input = tf.expand_dims(targ_emb[:, t, :], 1)

        perplexity_batch = tf.math.reduce_sum(tf.exp(rc_loss_sentence))/size
        perplexity += perplexity_batch

        avg_rc_loss = (rc_loss / size)
        avg_bow_loss = bow_loss_function(targs, bow_logit)
        # need to make kl divergence weight be 1
        elbo, aug_elbo, kl_loss = loss_function(avg_rc_loss, avg_bow_loss, recog_mu, recog_logvar, prior_mu,
                                                prior_logvar, kl_fullstep, kl_fullstep)
        rc_losses += avg_rc_loss
        elbo_losses += elbo
        bow_losses += avg_bow_loss
        kl_losses += kl_loss

        batch_num = k // batch_size + 1
        print('batch {} ELBO {:.4f} BOW LOSS {:.4f}  KL LOSS {:.4f} RC_Loss {:.4f} Perplexity {:.4f}'.format(batch_num, elbo, avg_bow_loss, kl_loss, avg_rc_loss, perplexity_batch))
        print('Time taken {} sec\n'.format(time.time() - start))

    if Have_small_batch:
        num_batch = num_batch + 1
    print('FINAL ELBO {:.4f} BOW LOSS {:.4f} KL LOSS {:.4f} RC LOSS {:.4f}'.format(elbo_losses / num_batch,
                                                                       bow_losses / num_batch,
                                                                       kl_losses / num_batch,
                                                                       rc_losses / num_batch))
    perplexity = perplexity/num_batch
    print("Average perplexity: {}".format(perplexity))

    print("Perplexity of average loss: {:.4f}".format(tf.exp(rc_losses / num_batch)))
