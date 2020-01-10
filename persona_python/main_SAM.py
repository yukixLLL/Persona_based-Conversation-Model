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
checkpoint_dir = '../persona_checkpoint'
if __name__ == "__main__":
    tf.debugging.set_log_device_placement(True)
    try:
        # Specify an invalid GPU device
        with tf.device('/device:GPU:1'):
            # loading
            with open(PATH + 'tokenizer.pickle', 'rb') as handle:
                tokenizer = pickle.load(handle)
            d_tensor_train = np.load(PATH + 'd_tensor_train.npy',allow_pickle=True)
            r_tensor_train = np.load(PATH + 'r_tensor_train.npy',allow_pickle=True)

            dia_train = [t[0] for t in d_tensor_train]
            aid_train = [t[1] for t in d_tensor_train]
            res_train = [t[0] for t in r_tensor_train]
            sid_train = [t[1] for t in r_tensor_train]

            BUFFER_SIZE = len(d_tensor_train)
            steps_per_epoch = len(d_tensor_train) // BATCH_SIZE + 1
            vocab_size = len(tokenizer.word_index) + 1

            # create tf.dataset
            dataset = tf.data.Dataset.from_tensor_slices((dia_train, res_train, sid_train, aid_train)).shuffle(BUFFER_SIZE)
            dataset = dataset.batch(BATCH_SIZE, drop_remainder=False)
            print("Create dataset done.")

            encoder = Encoder(HIDDEN_SIZE, vocab_size, embedding_dim, NUM_LAYER, BATCH_SIZE)
            decoder = Decoder(HIDDEN_SIZE, vocab_size, embedding_dim, speaker_dim,NUM_LAYER)
            optimizer = tf.keras.optimizers.Adam()

            train_nn_sa = Train(encoder, decoder, optimizer, tokenizer)     # speaker-addressee mode

            print("Start training")
            #################################### Training  ###########################################
            # speaker-addressee mode train
            speaker_add_cp_prefix = os.path.join(checkpoint_dir, "speaker-add-ckpt")
            speaker_add_cp = tf.train.Checkpoint(optimizer=train_nn_sa.optimizer,
                                                 encoder=train_nn_sa.encoder,
                                                 decoder=train_nn_sa.decoder)
            print("training in speaker-addressee mode")
            train_nn_sa.run_iter(EPOCHS, True, steps_per_epoch, dataset, speaker_add_cp, speaker_add_cp_prefix)

            ###########################################################################################

            backend.clear_session()

    except RuntimeError as e:
        print(e)