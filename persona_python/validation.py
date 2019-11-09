import tensorflow as tf
import numpy as np
import time
from helpers import count_real_word
from param import BATCH_SIZE, HIDDEN_SIZE

def validation(train_nn, inp, targ, speaker_id, addressee_id=None, batch_size=BATCH_SIZE):
    loss = 0
    inp = np.asarray(inp)
    targ = np.asarray(targ)
    speaker_id = np.asarray(speaker_id)
    if addressee_id is not None:
        addressee_id = np.asarray(addressee_id)
    #     print("test targ[:,1].shape: {}".format(targ[:,1].shape))
    val_size = targ.shape[0]
    print("val_size: {}".format(val_size))
    num_each_batch = (int)(np.floor(val_size / batch_size))
    remaining_num = val_size - num_each_batch * batch_size
    if remaining_num == 0:
        remaining_num = batch_size
    for k in range(0, val_size, batch_size):
        start = time.time()
        batch_loss = 0

        if (k + batch_size) >= val_size:
            print("k now +batch_size>=val_size: {}".format(k))
            inputs = inp[k:]
            targs = targ[k:]
            s_id = speaker_id[k:]
            #             print("s_id.shape: {}".format(s_id.shape))
            if addressee_id is not None:
                a_id = addressee_id[k:]
            enc_hidden = [tf.zeros((remaining_num, HIDDEN_SIZE)), tf.zeros((remaining_num, HIDDEN_SIZE))]

            enc_out, enc_hidden, enc_c = train_nn.encoder(inputs, enc_hidden)
            dec_init_state = [enc_hidden, enc_c]
            dec_input = tf.expand_dims([train_nn.tokenizer.word_index['<sos>']] * remaining_num, 1)
        else:
            print("k now: {}".format(k))
            inputs = inp[k:k + batch_size]
            targs = targ[k:k + batch_size]
            s_id = speaker_id[k:k + batch_size]
            # print("s_id.shape: {}".format(s_id.shape))
            if addressee_id is not None:
                a_id = addressee_id[k:k + batch_size]

            enc_hidden = [tf.zeros((batch_size, HIDDEN_SIZE)), tf.zeros((batch_size, HIDDEN_SIZE))]
            enc_out, enc_hidden, enc_c = train_nn.encoder(inputs, enc_hidden)
            dec_init_state = [enc_hidden, enc_c]
            dec_input = tf.expand_dims([train_nn.tokenizer.word_index['<sos>']] * batch_size, 1)

        word_per_line = count_real_word(targs)

        for t in range(targ.shape[1]):
            if addressee_id is not None:
                predictions, dec_hidden, dec_c, _ = train_nn.decoder(dec_input, enc_out, dec_init_state, s_id, a_id)
            else:
                predictions, dec_hidden, dec_c, _ = train_nn.decoder(dec_input, enc_out, dec_init_state, s_id)

            # use the max prob one in each sentence in the batch
            predicted_id = tf.argmax(predictions, axis=1)
            dec_init_state = [dec_hidden, dec_c]
            #             batch_loss += train_nn.loss_function(targs[:,t], predictions)
            loss_ = train_nn.loss_function(targs[:, t], predictions)
            batch_loss += tf.math.reduce_sum(loss_ / word_per_line)
            #             predicted = tf.constant(predicted_id)
            dec_input = tf.expand_dims(predicted_id, 1)

        if (k + batch_size) >= val_size:
            batch_loss = batch_loss / remaining_num
        else:
            batch_loss = batch_loss / batch_size

        loss += batch_loss
        batch_num = k / batch_size + 1
        print('batch {} Loss {:.4f}'.format(batch_num,
                                            batch_loss))
        print('Time taken {} sec\n'.format(time.time() - start))
    return loss/batch_num