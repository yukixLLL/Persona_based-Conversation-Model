import tensorflow as tf
from param import *
from helpers import *
from encoder import Encoder
from decoder import Decoder
from train import Train
from validation import validation
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras import backend

BATCH_SIZE = 96 # on cpu, 96 on gpu
EPOCHS = 3
PATH = '../data/'
checkpoint_dir = './persona_checkpoint'
if __name__ == "__main__":
    tf.debugging.set_log_device_placement(True)
    try:
        # Specify an invalid GPU device
        with tf.device('/device:GPU:0'):
            # print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

            # data_size, speakerid_list, speakers, dialogues, episodes = load_dataset(PATH)
            # tensor, tokenizer, sent_list = tokenize(dialogues, data_size)
            # d_tensor, r_tensor = create_dataset(tensor, episodes, speakerid_list, data_size)
            #
            # # shuffle
            # tensor = shuffle(d_tensor, r_tensor)
            # d_tensor = tensor[0]
            # r_tensor = tensor[1]
            #
            # # Creating training and validation sets using an 80-20 split
            # d_tensor_train, d_tensor_val, r_tensor_train, r_tensor_val = train_test_split(d_tensor, r_tensor, test_size=0.2)
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
            example_input_batch, example_target_batch, example_sid_batch, example_aid_batch = next(iter(dataset))
            print("Create dataset done.")

            encoder = Encoder(HIDDEN_SIZE, vocab_size, embedding_dim, NUM_LAYER, BATCH_SIZE)
            decoder = Decoder(HIDDEN_SIZE, vocab_size, embedding_dim, NUM_LAYER)
            optimizer = tf.keras.optimizers.Adam()

            cp = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)
            status = cp.restore("persona_checkpoint/combination-ckpt-5")
            print(status)

            train_nn = Train(encoder, decoder, optimizer, tokenizer)


            print("Start training")
            #################################### test Way ###########################################
            # speaker mode test-train
            # print("training in speaker mode")
            # train_nn.run_iter_test(EPOCHS, False, steps_per_epoch, dataset)

            # speaker-addressee mode test-train
            print("training in speaker-addressee mode")
            train_nn.run_iter_test(EPOCHS, True, steps_per_epoch, dataset)

            batch_sz = 16
            size = 2 * batch_sz + 5
            sample_input_val = dia_val[:size]
            sample_targ_val = res_val[:size]
            sample_sid_val = sid_val[:size]
            sample_aid_val = aid_val[:size]

            # speaker model
            print("validation in speaker mode")
            s_loss = validation(train_nn, sample_input_val, sample_targ_val, sample_sid_val, batch_size=batch_sz)
            print("loss in speaker mode: {:.4f}".format(s_loss.numpy()))

            # speaker-addressee model
            print("validation in speaker-addressee mode")
            sa_loss = validation(train_nn, sample_input_val, sample_targ_val, sample_sid_val, sample_aid_val,
                                 batch_size=batch_sz)
            print("loss in speaker-addressee mode: {:.4f}".format(sa_loss.numpy()))
            ###########################################################################################

            backend.clear_session()


    except RuntimeError as e:
        print(e)