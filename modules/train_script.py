from sklearn.model_selection import train_test_split
import argparse
import numpy as np
import torch
from modules.preprocess import text_edit
from modules.utils import build_dataset, text_to_word2vec, zip_set, euclid_dis, eucl_dist_output_shape, contrastive_loss, accuracy
from modules.model import create_base_net_1D
import spacy
from gensim.models import KeyedVectors
from keras.layers import Lambda, Input
from keras.models import Model
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint

parser = argparse.ArgumentParser()
parser.add_argument("num_samples", type=int)
parser.add_argument("batch_size", type=int)
parser.add_argument("epochs", type=int)
parser.add_argument("max_len", type=int)
args = parser.parse_args()


if __name__ == "__main__":

    nlp = spacy.load("fr_core_news_sm")

    dataset = build_dataset(path="siamese_net/data",num_samples=args.num_samples, max_len=args.max_len, rnd_state=10)

    dataset = text_edit(dataset,grp_num=False,rm_newline=True,rm_punctuation=True,lowercase=True,lemmatize=False,html_=False,convert_entities=False,expand=False)

    X = [x['text'] for x in dataset.values() if x['section_1'] in ['actualites', 'sports', 'affaires', 'arts', 'international']]
    Y = [x['section_label'] for x in dataset.values() if x['section_1'] in ['actualites', 'sports', 'affaires', 'arts', 'international']]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    model_path = 'wiki.fr.vec'
    word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    text = "Ceci est un texte exemple"
    vector = text_to_word2vec(text, word2vec_model)
    shape = vector.shape[0]

    X_train = torch.stack([torch.tensor(text_to_word2vec(x, word2vec_model), dtype=torch.float32) for x in X_train], dim=0)
    X_test = torch.stack([torch.tensor(text_to_word2vec(x, word2vec_model), dtype=torch.float32) for x in X_test], dim=0)
    Y_train = torch.tensor(Y_train, dtype=torch.long)
    Y_test = torch.tensor(Y_test, dtype=torch.long)

    tp =  args.num_samples * 5
    tt =  args.num_samples * 1

    train_pairs, train_labels = zip_set(X_train, Y_train, num_pairs=tp)

    test_pairs, test_labels = zip_set(X_test, Y_test, num_pairs=tt)

    base_network  = create_base_net_1D((shape,args.max_len))


    input_a = Input(shape=(shape,args.max_len))
    input_b = Input(shape=(shape,args.max_len))

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(euclid_dis,output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    model = Model([input_a, input_b], distance)

    model_checkpoint_callback = ModelCheckpoint(
        filepath='path/to/save/model',  
        save_weights_only=False,  
        monitor='val_accuracy',  
        mode='max',  
        save_best_only=True) 

    epochs=args.epochs
    rms = RMSprop()
    model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
    model.fit([train_pairs[:, 0], train_pairs[:, 1]], train_labels,
            batch_size=args.batch_size,
            epochs=epochs,
            validation_data=([test_pairs[:, 0], test_pairs[:, 1]], test_labels),
            callbacks=[model_checkpoint_callback])