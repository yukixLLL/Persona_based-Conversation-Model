import tensorflow as tf
from util import *
import time
from params import BATCH_SIZE,Use_Prior


class Train(object):
    def __init__(self, word_embedding, encoder, decoder, recognition_network, prior_network, generation_network,
                 optimizer, tokenizer):
        self.word_embedding = word_embedding
        self.encoder = encoder
        self.decoder = decoder
        self.recognition_network = recognition_network
        self.prior_network = prior_network
        self.generation_network = generation_network
        self.tokenizer = tokenizer
        self.optimizer = optimizer

    def distributed_train_step(self,strategy, inp, targ, global_iter, kl_full_step, speaker_id, batch_size, addressee_id=None):
        # define training step
        def train_step(inp, targ, global_iter, kl_full_step, speaker_id, batch_sz=BATCH_SIZE, addressee_id=None):
            word_per_line = count_real_word(targ)
            with tf.GradientTape() as tape:
                inp_emb = self.word_embedding(inp)
                targ_emb = self.word_embedding(targ)
                enc_output, enc_hidden, enc_c = self.encoder(inp_emb)
                _, targ_hidden, _ = self.encoder(targ_emb)
                # VAE module
                prior_mu, prior_logvar = self.prior_network(enc_hidden)
                recog_mu, recog_logvar = self.recognition_network(enc_hidden, targ_hidden)

                latent_sample = tf.cond(Use_Prior, lambda: sample_gaussian(prior_mu, prior_logvar),
                                        lambda: sample_gaussian(recog_mu, recog_logvar))
                dec_init_state, bow_logit = self.generation_network(enc_hidden, latent_sample)

                #             dec_input = self.word_embedding(tf.expand_dims([self.tokenizer.word_index['<sos>']]*batch_size,1))
                dec_input = tf.expand_dims(targ_emb[:, 0, :], 1)
                rc_loss = 0
                # Teacher forcing - feeding the target as the next input
                for t in range(1, targ.shape[1]):
                    # passing enc_output to the decoder
                    if addressee_id is not None:
                        predictions, dec_hidden, dec_c, _ = self.decoder(
                            dec_input, enc_output, dec_init_state, speaker_id, addressee_id)
                    else:
                        predictions, dec_hidden, dec_c, _ = self.decoder(
                            dec_input, enc_output, dec_init_state, speaker_id)
                    dec_init_state = [dec_hidden, dec_c]
                    rc_loss += rc_loss_function(targ[:, t], predictions, word_per_line)

                    # using teacher forcing
                    #                 dec_input = self.word_embedding(tf.expand_dims(targ[:, t], 1))
                    dec_input = tf.expand_dims(targ_emb[:, t, :], 1)

                avg_rc_loss = (rc_loss / batch_sz)
                avg_bow_loss = bow_loss_function(targ, bow_logit,batch_sz)
                elbo, aug_elbo, kl_loss = loss_function(avg_rc_loss, avg_bow_loss, recog_mu, recog_logvar, prior_mu,
                                                        prior_logvar, batch_sz,global_iter, kl_full_step)
                variables = self.word_embedding.trainable_variables + self.encoder.trainable_variables + self.decoder.trainable_variables + self.recognition_network.trainable_variables + self.prior_network.trainable_variables + self.generation_network.trainable_variables
                gradients = tape.gradient(aug_elbo, variables)
                self.optimizer.apply_gradients(zip(gradients, variables))
                return elbo, avg_bow_loss, avg_rc_loss, kl_loss
        
        elbo,avg_bow_loss,avg_rc_loss,kl_loss = strategy.experimental_run_v2(train_step,
                                                      args=(inp, targ, global_iter, kl_full_step, speaker_id, batch_size, addressee_id,))
        print("elbo.shape: {}".format(elbo.shape))
        print("avg_bow_loss.shape: {}".format(avg_bow_loss.shape))
        print("avg_rc_loss.shape: {}".format(avg_rc_loss.shape))
        print("kl_loss.shape: {}".format(kl_loss.shape))
        dis_elbo = strategy.reduce(tf.distribute.ReduceOp.SUM, elbo,axis=None)
        dis_avg_bow_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, avg_bow_loss,axis=None)
        dis_avg_rc_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, avg_rc_loss,axis=None)
        dis_kl_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, kl_loss,axis=None)

        return dis_elbo,dis_avg_bow_loss,dis_avg_rc_loss,dis_kl_loss

    def run_iter(self, strategy, epochs, isAddressee, steps_per_epoch, dataset, checkpoint, checkpoint_prefix, batch_sz = BATCH_SIZE):
        global_iter = 0
        kl_full_step = steps_per_epoch * epochs / 2
        batch_num = steps_per_epoch
        for e in range(epochs):
            start = time.time()
            elbo_losses = 0
            bow_losses = 0
            rc_losses = 0
            kl_losses = 0
            for (batch, (inp, targ, sid, aid)) in enumerate(dataset):
                if isAddressee == True:
                    # global iter to count iter
                    elbo, avg_bow_loss, avg_rc_loss, kl_loss = self.distributed_train_step(strategy,inp, targ, global_iter, kl_full_step,
                                                                               sid, batch_sz, aid)
                else:
                    # global iter to count iter
                    elbo, avg_bow_loss, avg_rc_loss, kl_loss = self.distributed_train_step(strategy,inp, targ, global_iter, kl_full_step,
                                                                               sid, batch_sz)
                elbo_losses += elbo
                bow_losses += avg_bow_loss
                rc_losses += avg_rc_loss
                kl_losses += kl_loss
                global_iter += 1
                if batch % 100 == 0:
                    print('Epoch {} Batch {} ELBO {:.4f} BOW LOSS {:.4f} RC LOSS {:.4f} KL LOSS {:.4f}'.format(e + 1,
                                                                                                               batch,
                                                                                                               elbo.numpy(),
                                                                                                               avg_bow_loss.numpy(),
                                                                                                               avg_rc_loss.numpy(),
                                                                                                               kl_loss.numpy()))

            # saving (checkpoint) the model every 2 epochs
            if (e + 1) % 2 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)

            print('Epoch {} ELBO {:.4f} BOW LOSS {:.4f} RC LOSS {:.4f}'.format(e + 1,
                                                                               elbo_losses / batch_num,
                                                                               bow_losses / batch_num,
                                                                               rc_losses / batch_num,
                                                                               kl_losses / batch_num))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    def run_iter_test(self,strategy, epochs, isAddressee, steps_per_epoch, dataset,batch_sz = BATCH_SIZE):
        global_iter = 0
        kl_full_step = steps_per_epoch * epochs / 2
        batch_num = steps_per_epoch
        for e in range(epochs):
            start = time.time()
            elbo_losses = 0
            bow_losses = 0
            rc_losses = 0
            kl_losses = 0
            for (batch, (inp, targ, sid, aid)) in enumerate(dataset):
                # batch_sz = targ.shape[0]
                print("batch size:{}".format(batch_sz))
                if isAddressee == True:
                    # global iter to count iter
                    elbo, avg_bow_loss, avg_rc_loss, kl_loss = self.distributed_train_step(strategy,inp, targ, global_iter, kl_full_step,
                                                                               sid, batch_num, aid)
                else:
                    # global iter to count iter
                    elbo, avg_bow_loss, avg_rc_loss, kl_loss = self.distributed_train_step(strategy,inp, targ, global_iter, kl_full_step,
                                                                               sid, batch_num)
                elbo_losses += elbo
                bow_losses += avg_bow_loss
                rc_losses += avg_rc_loss
                kl_losses += kl_loss
                global_iter += 1

                print('Epoch {} Batch {} ELBO {:.4f} BOW LOSS {:.4f} RC LOSS {:.4f} KL LOSS {:.4f}'.format(e + 1,
                                                                                                               batch,
                                                                                                               elbo.numpy(),
                                                                                                               avg_bow_loss.numpy(),
                                                                                                               avg_rc_loss.numpy(),
                                                                                                               kl_loss.numpy()))

            print('Epoch {} ELBO {:.4f} BOW LOSS {:.4f} RC LOSS {:.4f}'.format(e + 1,
                                                                               elbo_losses / batch_num,
                                                                               bow_losses / batch_num,
                                                                               rc_losses / batch_num,
                                                                               kl_losses / batch_num))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))