import numpy as np
from params import *
from word_embedding import *
from encoder import *
from decoder import *
from recognition_nw import *
from prior_nw import *
from generation_nw import *
from train import *
from validation import *
import pickle
from tensorflow.keras import backend

PATH = '../data/'
checkpoint_dir = '../vae_ckpt_SAM_400enc_512dec_200latent/'

if __name__ == "__main__":

    with open(PATH + 'tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    d_tensor_val = np.load(PATH + 'd_tensor_val.npy', allow_pickle=True)
    r_tensor_val = np.load(PATH + 'r_tensor_val.npy', allow_pickle=True)

    dia_val = [t[0] for t in d_tensor_val]
    aid_val = [t[1] for t in d_tensor_val]
    res_val = [t[0] for t in r_tensor_val]
    sid_val = [t[1] for t in r_tensor_val]

    vocab_size = len(tokenizer.word_index) + 1

    word_embedding = Word_Embedding(vocab_size, embedding_dim)
    encoder = Encoder(ENC_HIDDEN_SIZE, vocab_size, BATCH_SIZE)
    decoder = Decoder(DEC_HIDDEN_SIZE, vocab_size, speaker_dim, 4)
    recognition_network = Recognition_Network(LATENT_SIZE)
    prior_network = Prior_Network(LATENT_SIZE)
    generation_network = Generation_Network(vocab_size, DEC_HIDDEN_SIZE)
    optimizer = tf.keras.optimizers.Adam()

    cp = tf.train.Checkpoint(optimizer=optimizer,
                             word_embedding=word_embedding,
                             encoder=encoder,
                             decoder=decoder,
                             recognition_network=recognition_network,
                             prior_network=prior_network,
                             generation_network=generation_network)
    status = cp.restore(checkpoint_dir + "speaker-add-ckpt-3")
    print(status)
    train_nn = Train(word_embedding, encoder, decoder, recognition_network, prior_network, generation_network,optimizer,tokenizer)

    # speaker mode validation
    # print("validation in speaker mode")
    # validation(train_nn, dia_val, res_val, sid_val, batch_size=128)

    # speaker-addressee mode validation
    print("validation in speaker-addressee mode")
    validation(train_nn, dia_val, res_val, sid_val, aid_val,batch_size=128)

    backend.clear_session()

