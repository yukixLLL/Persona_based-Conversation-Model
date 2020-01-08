import tensorflow as tf
import os
import pickle
import numpy as np
from params import *
from word_embedding import *
from encoder import *
from decoder import *
from recognition_nw import *
from prior_nw import *
from generation_nw import *
from train import *
from tensorflow.keras import backend

PATH = '../data/'
checkpoint_dir = '../vae_checkpoint'

def main():
    with open(PATH + 'tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    d_tensor_train = np.load(PATH + 'd_tensor_train.npy', allow_pickle=True)
    r_tensor_train = np.load(PATH + 'r_tensor_train.npy', allow_pickle=True)

    dia_train = [t[0] for t in d_tensor_train]
    aid_train = [t[1] for t in d_tensor_train]
    res_train = [t[0] for t in r_tensor_train]
    sid_train = [t[1] for t in r_tensor_train]

    BUFFER_SIZE = len(d_tensor_train)
    steps_per_epoch = int(np.ceil(len(d_tensor_train) / BATCH_SIZE))
    vocab_size = len(tokenizer.word_index) + 1

    # create tf.dataset
    dataset = tf.data.Dataset.from_tensor_slices((dia_train, res_train, sid_train, aid_train)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=False)
    print("Create dataset done.")

    #################################### INITIALIZATION ###########################################
    word_embedding = Word_Embedding(vocab_size, embedding_dim)
    encoder = Encoder(ENC_HIDDEN_SIZE, vocab_size, BATCH_SIZE)
    decoder = Decoder(DEC_HIDDEN_SIZE, vocab_size, speaker_dim, NUM_LAYER)
    recognition_network = Recognition_Network(LATENT_SIZE)
    prior_network = Prior_Network(LATENT_SIZE)
    generation_network = Generation_Network(vocab_size, DEC_HIDDEN_SIZE)
    optimizer = tf.keras.optimizers.Adam()

    train_nn = Train(word_embedding, encoder, decoder, recognition_network, prior_network, generation_network,
                     optimizer, tokenizer)

    #################################### TRAINING ###########################################
    print("Start training")
    # speaker mode training
    speaker_cp_prefix = os.path.join(checkpoint_dir, "speaker-ckpt")
    speaker_cp = tf.train.Checkpoint(optimizer=train_nn.optimizer,
                                     word_embedding = train_nn.word_embedding,
                                     encoder=train_nn.encoder,
                                     decoder=train_nn.decoder,
                                     recognition_network = train_nn.recognition_network,
                                     prior_network = train_nn.prior_network,
                                     generation_network = train_nn.generation_network)
    print("training in speaker mode")
    train_nn.run_iter(EPOCHS, False, steps_per_epoch, dataset, speaker_cp, speaker_cp_prefix)


if __name__ == "__main__":
    tf.debugging.set_log_device_placement(True)
    try:
        # Specify an invalid GPU device
        with tf.device('/device:GPU:0'):
            main()
            backend.clear_session()
    except RuntimeError as e:
        print(e)