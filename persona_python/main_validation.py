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
checkpoint_dir = '../persona_ckpt_400_sp128/'

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

    encoder = Encoder(HIDDEN_SIZE, vocab_size, embedding_dim, NUM_LAYER, BATCH_SIZE)
    decoder = Decoder(HIDDEN_SIZE, vocab_size, embedding_dim,speaker_dim, NUM_LAYER)
    optimizer = tf.keras.optimizers.Adam()

    cp = tf.train.Checkpoint(optimizer=optimizer,
                             encoder=encoder,
                             decoder=decoder)
    status = cp.restore(checkpoint_dir + "speaker-ckpt-5")
    print(status)
    train_nn = Train(encoder, decoder, optimizer, tokenizer)

    # speaker mode validation
    print("validation in speaker mode")
    s_loss = validation(train_nn, dia_val, res_val, sid_val, batch_size=128)
    print("loss in speaker mode: {:.4f}".format(s_loss.numpy()))
    print("perplexity of average loss:{:.4f}".format(tf.exp(s_loss).numpy()))

    # speaker-addressee mode validation
    # print("validation in speaker-addressee mode")
    # sa_loss = validation(train_nn, dia_val, res_val, sid_val, aid_val,batch_size=128)
    # print("loss in speaker-addressee mode: {:.4f}".format(sa_loss.numpy()))
    # print("perplexity of average loss:{:.4f}".format(tf.exp(sa_loss).numpy()))

    backend.clear_session()
