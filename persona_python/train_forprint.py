import tensorflow as tf
import time
from helpers import count_real_word
from param import BATCH_SIZE
class Train(object):
    def __init__(self, encoder, decoder, optimizer, tokenizer):
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')

    def loss_function(self, real, pred):
        # loss of every word
        # true word mask
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ = loss_ * mask
        return loss_

    #     @tf.function
    def train_step(self, inp, targ, enc_hidden, speaker_id, batch_size=BATCH_SIZE, addressee_id=None):
        loss = 0
        word_per_line = count_real_word(targ)
        with tf.GradientTape() as tape:
            enc_output, enc_hidden, enc_c = self.encoder(inp, enc_hidden)
            dec_init_state = [enc_hidden, enc_c]
            dec_input = tf.expand_dims([self.tokenizer.word_index['<sos>']] * batch_size, 1)

            # Teacher forcing - feeding the target as the next input
            for t in range(1, targ.shape[1]):
                # passing enc_output to the decoder
                if addressee_id is not None:
                    #                     print("detect addressee")
                    predictions, dec_hidden, dec_c, _ = self.decoder(
                        dec_input, enc_output, dec_init_state, speaker_id, addressee_id)
                else:
                    predictions, dec_hidden, dec_c, _ = self.decoder(
                        dec_input, enc_output, dec_init_state, speaker_id)
                dec_init_state = [dec_hidden, dec_c]
                #                 loss += self.loss_function(targ[:,t], predictions)
                loss_ = self.loss_function(targ[:, t], predictions)
                loss += tf.math.reduce_sum(loss_ / word_per_line)

                # using teacher forcing
                dec_input = tf.expand_dims(targ[:, t], 1)

            # batch_size = targ.shape[0]
            batch_loss = (loss / int(targ.shape[0]))
            variables = self.encoder.trainable_variables + self.decoder.trainable_variables

            gradients = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(gradients, variables))
            return batch_loss

    def run_iter(self, epochs, isAddressee, steps_per_epoch, dataset, checkpoint, checkpoint_prefix):
        for e in range(epochs):
            start = time.time()
            total_loss = 0

            for (batch, (inp, targ, sid, aid)) in enumerate(dataset.take(steps_per_epoch)):
                # drop_reminder = False
                batch_sz = targ.shape[0]

                enc_hidden = self.encoder.initialize_hidden_state(batch_sz)
                if isAddressee == True:
                    batch_loss = self.train_step(inp, targ, enc_hidden, sid, batch_sz, aid)
                else:
                    batch_loss = self.train_step(inp, targ, enc_hidden, sid, batch_sz)
                total_loss += batch_loss

                # print every branch
                print('Epoch {} Batch {} Loss {:.4f}'.format(e + 1,
                                                                 batch,
                                                                 batch_loss.numpy()))

            # saving (checkpoint) the model every 2 epochs
            if (e + 1) % 2 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)

            print('Epoch {} Loss {:.4f}'.format(e + 1, total_loss / steps_per_epoch))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    def run_iter_test(self, epochs, isAddressee, steps_per_epoch, dataset):
        for e in range(epochs):
            start = time.time()
            total_loss = 0

            for (batch, (inp, targ, sid, aid)) in enumerate(dataset.take(steps_per_epoch)):
                # drop_reminder = False
                batch_sz = targ.shape[0]

                enc_hidden = self.encoder.initialize_hidden_state(batch_sz)
                if isAddressee == True:
                    batch_loss = self.train_step(inp, targ, enc_hidden, sid, batch_sz, aid)
                else:
                    batch_loss = self.train_step(inp, targ, enc_hidden, sid, batch_sz)
                total_loss += batch_loss

                # just for test
                print('Epoch {} Batch {} Loss {:.4f}'.format(e + 1,
                                                             batch,
                                                             batch_loss.numpy()))
                # just for test
                if batch == 3: break

            # just for test
            print('Epoch {} Loss {:.4f}'.format(e + 1,
                                                total_loss / 3))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))