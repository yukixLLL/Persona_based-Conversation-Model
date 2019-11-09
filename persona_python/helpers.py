import unicodedata
import re
import os
import io
import tensorflow as tf
import pandas as pd
from param import MAXLEN


# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.rstrip().strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<SOS> ' + w + ' <EOS>'
    return w

def load_dataset(path,friends_='friends.csv',bigbang_='bigbang.csv'):
    friends_df = pd.read_csv(path + friends_)
    bigbang_df = pd.read_csv(path + bigbang_)

    # need to drop sentence which are NA (because they represents some action of the characters)
    na_index = bigbang_df[bigbang_df['dialogue'].isna()].index
    bigbang_df.drop(index=na_index, inplace=True)

    df = pd.concat([friends_df, bigbang_df], ignore_index=True, sort=False)
    df.reset_index(drop=True, inplace=True)
    main_c = ['joey', 'rachel', 'chandler', 'monica', 'ross', 'phoebe', 'leonard',
              'sheldon', 'penny', 'howard', 'raj', 'amy', 'bernadette', 'other']
    speakers_ind = dict()
    for ind, c in enumerate(main_c, 1):
        speakers_ind[c] = ind

    df['speaker_id'] = df['speakers'].apply(lambda x: speakers_ind[x] - 1)
    speakerid_list = list(df['speaker_id'])
    speakers = list(df['speakers'])
    dialogues = list(df['dialogue'])
    episodes = list(df['episodes'])
    data_size = len(df)
    return data_size, speakerid_list, speakers, dialogues, episodes


def max_length(tensor):
    return max(len(t[0]) for t in tensor)

def tokenize(dialogues,num_samples):
    sent_list = list()
    for k in range(0,num_samples):
        sent_list.append(preprocess_sentence(dialogues[k]))
    sent_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    sent_tokenizer.fit_on_texts(sent_list)

    tensor = sent_tokenizer.texts_to_sequences(sent_list)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, maxlen=MAXLEN, padding='post', truncating='post', value=0)

    return tensor, sent_tokenizer,sent_list

def create_dataset(tensor,episodes,speakerid_list,num_samples):
    dialogues_list = list()
    response_list = list()
    for k in range(0,num_samples):
        if(k+1 >= num_samples):
            break
        if episodes[k]==episodes[k+1]:
            dialogue = tensor[k]
#             pdb.set_trace()
            response = tensor[k+1]
            addressee = tf.convert_to_tensor(speakerid_list[k])
            speaker = tf.convert_to_tensor(speakerid_list[k+1])
            dialogues_list.append([dialogue,addressee])
            response_list.append([response,speaker])
    return dialogues_list,response_list



def count_real_word(real):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    mask = tf.cast(mask, dtype=tf.float32)
    word_per_line = tf.math.reduce_sum(mask,1)
    return word_per_line