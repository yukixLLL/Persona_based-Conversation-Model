import tensorflow as tf
import numpy as np
import time
from helpers import count_real_word
from param import BATCH_SIZE, HIDDEN_SIZE

def validation(train_nn, inp, targ, speaker_id, addressee_id=None, batch_size=BATCH_SIZE):
    loss = 0
    perplexity = 0
    inp = np.asarray(inp)
    targ = np.asarray(targ)
    speaker_id = np.asarray(speaker_id)
    if addressee_id is not None:
        addressee_id = np.asarray(addressee_id)
    val_size = targ.shape[0]
    # print("val_size: {}".format(val_size))
    num_batch = (int)(np.floor(val_size / batch_size))
    remaining_num = val_size - num_batch * batch_size
    if remaining_num == 0:
        remaining_num = batch_size
    for k in range(0, val_size, batch_size):
        start = time.time()
        batch_loss = 0
        size = batch_size
        if (k + batch_size) >= val_size:
            size = remaining_num
        inputs = inp[k:k + size]
        targs = targ[k:k + size]
        s_id = speaker_id[k:k + size]
        if addressee_id is not None:
            a_id = addressee_id[k:k + size]

        enc_hidden = [tf.zeros((size, HIDDEN_SIZE)), tf.zeros((size, HIDDEN_SIZE))]
        enc_out, enc_hidden, enc_c = train_nn.encoder(inputs, enc_hidden)
        dec_init_state = [enc_hidden, enc_c]
        dec_input = tf.expand_dims([train_nn.tokenizer.word_index['<sos>']] * size, 1)

        word_per_line = count_real_word(targs)
        loss_sentence = 0
        for t in range(1, targs.shape[1]):
            if addressee_id is not None:
                predictions, dec_hidden, dec_c, _ = train_nn.decoder(dec_input, enc_out, dec_init_state, s_id, a_id)
            else:
                predictions, dec_hidden, dec_c, _ = train_nn.decoder(dec_input, enc_out, dec_init_state, s_id)

            # use the max prob one in each sentence in the batch
            # predicted_id = tf.argmax(predictions, axis=1)
            dec_init_state = [dec_hidden, dec_c]
            loss_ = train_nn.loss_function(targs[:, t], predictions)
            loss_sentence += loss_ / word_per_line
            batch_loss += tf.math.reduce_sum(loss_ / word_per_line)
            dec_input = tf.expand_dims(targs[:, t], 1)

        # perplexity of each sentence
        perplexity_batch = tf.math.reduce_sum(tf.exp(loss_sentence))
        perplexity += perplexity_batch
        loss += batch_loss
        if(batch_size>1):
            batch_num = k / batch_size + 1
            print('batch {} Loss {:.4f} Perplexity {:.4f}'.format(batch_num,
                                                batch_loss/size, perplexity_batch/size))
            print('Time taken {} sec\n'.format(time.time() - start))
        else:
            batch_num = val_size
            if ((k + 1) % 128 == 0):
                print('{}th senntence now, current batch loss {:.4f}, current loss {:.4f}, current perplexity:{:4f}'.format(k + 1, batch_loss/size,
                                                    loss/(k+1), tf.exp(batch_loss/size)))

    # perplexity using average loss
    perplexity = perplexity/val_size
    print("Perplexity: {}".format(perplexity))

    return loss/val_size