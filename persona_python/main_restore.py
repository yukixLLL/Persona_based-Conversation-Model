import tensorflow as tf
from param import *
from helpers import *
from encoder import Encoder
from decoder import Decoder
from train import Train
from validation import validation
import numpy as np
import pickle
from tensorflow.keras import backend

PATH = '../data/'
checkpoint_dir = './persona_checkpoint'
if __name__ == "__main__":
    tf.debugging.set_log_device_placement(True)
    try:
        # Specify an invalid GPU device
        with tf.device('/device:GPU:0'):
            # loading
            with open(PATH + 'tokenizer.pickle', 'rb') as handle:
                tokenizer = pickle.load(handle)
            d_tensor_train = np.load(PATH + 'd_tensor_train.npy',allow_pickle=True)
            d_tensor_val = np.load(PATH + 'd_tensor_val.npy',allow_pickle=True)
            r_tensor_train = np.load(PATH + 'r_tensor_train.npy',allow_pickle=True)
            r_tensor_val = np.load(PATH + 'r_tensor_val.npy',allow_pickle=True)

            dia_train = [t[0] for t in d_tensor_train]
            dia_val = [t[0] for t in d_tensor_val]
            aid_train = [t[1] for t in d_tensor_train]
            aid_val = [t[1] for t in d_tensor_val]
            res_train = [t[0] for t in r_tensor_train]
            res_val = [t[0] for t in r_tensor_val]
            sid_train = [t[1] for t in r_tensor_train]
            sid_val = [t[1] for t in r_tensor_val]

            BUFFER_SIZE = len(d_tensor_train)
            steps_per_epoch = len(d_tensor_train) // BATCH_SIZE + 1
            vocab_size = len(tokenizer.word_index) + 1

            # create tf.dataset
            dataset = tf.data.Dataset.from_tensor_slices((dia_train, res_train, sid_train, aid_train)).shuffle(BUFFER_SIZE)
            dataset = dataset.batch(BATCH_SIZE, drop_remainder=False)
            print("Create dataset done.")

            encoder = Encoder(HIDDEN_SIZE, vocab_size, embedding_dim, NUM_LAYER, BATCH_SIZE)
            decoder = Decoder(HIDDEN_SIZE, vocab_size, embedding_dim, NUM_LAYER)
            optimizer = tf.keras.optimizers.Adam()

            cp = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)
            status = cp.restore("persona_checkpoint_fst/speaker-add-ckpt-5")
            print(status)
            train_nn_sa = Train(encoder, decoder, optimizer, tokenizer)     # speaker-addressee mode restore


            # train_nn_s = Train(encoder, decoder, optimizer, tokenizer)  # speaker mode restore

            print("Start training")
            #################################### Formal Way ###########################################
            # speaker mode train
            # speaker_cp_prefix = os.path.join(checkpoint_dir, "speaker-ckpt")
            # speaker_cp = tf.train.Checkpoint(optimizer=train_nn_s.optimizer,
            #                                  encoder=train_nn_s.encoder,
            #                                  decoder=train_nn_s.decoder)
            # print("training in speaker mode")
            # train_nn_s.run_iter(EPOCHS, False, steps_per_epoch, dataset, speaker_cp, speaker_cp_prefix)

            # speaker mode validation
            # print("validation in speaker mode")
            # s_loss = validation(train_nn_s, dia_val, res_val, sid_val)
            # print("loss in speaker mode: {:.4f}".format(s_loss.numpy()))

            # speaker-addressee mode train
            # speaker_add_cp_prefix = os.path.join(checkpoint_dir, "speaker-add-ckpt")
            # speaker_add_cp = tf.train.Checkpoint(optimizer=train_nn_sa.optimizer,
            #                                      encoder=train_nn_sa.encoder,
            #                                      decoder=train_nn_sa.decoder)
            # print("training in speaker-addressee mode")
            # train_nn_sa.run_iter(EPOCHS, True, steps_per_epoch, dataset, speaker_add_cp, speaker_add_cp_prefix)



            # speaker-addressee mode validation
            print("validation in speaker-addressee mode")
            sa_loss = validation(train_nn_sa, dia_val, res_val, sid_val, aid_val)
            print("loss in speaker-addressee mode: {:.4f}".format(sa_loss.numpy()))
            ###########################################################################################

            backend.clear_session()

    except RuntimeError as e:
        print(e)