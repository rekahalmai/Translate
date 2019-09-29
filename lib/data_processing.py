import spacy
import re
from torchtext import data
from lib.batch import MyIterator, batch_size_fn
import numpy as np 
import pandas as pd
import os
import dill as pickle

def read_data(params):
    if params['src_path'] is not None:
        try:
            params['src_data'] = open(params['src_path']).read().strip().split('\n')
        except:
            print(f"Error: {params['src_path']} not found.")

    if params['trg_path'] is not None:
        try:
            params['trg_data'] = open(params['trg_path']).read().strip().split('\n')
        except:
            print("Error: {params['trg_path']} not found.")


class Tokenize:
    def __init__(self, lang):
        self.nlp = spacy.load(lang)

    def tokenizer(self, sentence):
        sentence = sentence.replace("\u202f", " ")
        sentence = re.sub(
            r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(sentence))
        sentence = re.sub(r"[ ]+", " ", sentence)
        sentence = re.sub(r"\!+", "!", sentence)
        sentence = re.sub(r"\,+", ",", sentence)
        sentence = re.sub(r"\?+", "?", sentence)
        sentence = sentence.lower()

        return [tok.text for tok in self.nlp.tokenizer(sentence) if tok.text != " "]


def create_fields(params):
    print("loading spacy tokenizers...")

    t_src = Tokenize(params["src_lang"])
    t_trg = Tokenize(params["trg_lang"])

    print("creating SRC and TRG...")
    TRG = data.Field(lower=True, tokenize=t_trg.tokenizer, init_token='<sos>', eos_token='<eos>')
    SRC = data.Field(lower=True, tokenize=t_src.tokenizer)

    if params["load_weights"] is not None:
        try:
            print("loading presaved fields...")
            SRC = pickle.load(open(f'{params["load_weights"]}/SRC.pkl', 'rb'))
            TRG = pickle.load(open(f'{params["load_weights"]}/TRG.pkl', 'rb'))
        except:
            print("error opening SRC.pkl and TXT.pkl field files, please ensure they are in " + params["load_weights"] + "/")
                               
            #quit()            
                                   
    return SRC, TRG


def create_dataset(params, SRC, TRG):
                                                  
    print("creating dataset and iterator... ")
                                   
    # Get the data in a dataframe
    raw_data = {'scr': [line for line in params['src_data']], 'trg': [line for line in params['trg_data']]}
    df = pd.DataFrame(raw_data, columns=["scr", "trg"])

    # get only the examples that have less than params["max_length"] characters
    mask = (df['scr'].str.count(' ') < params["max_length"]) & (df['trg'].str.count(' ') < params['max_length'])
    df = df.loc[mask]
                                                   
    # Save as a csv
    df.to_csv("data/translate_transformer_temp.csv", index=False)

    # Data fields - needed to create a TabularDataset
    data_fields = [('src', SRC), ('trg', TRG)]
    train = data.TabularDataset('./data/translate_transformer_temp.csv', format='csv', fields=data_fields)
    #os.remove('data/translate_transformer_temp.csv')

    # Create the vocab if not yet created
    if params["load_weights"] is None:
        SRC.build_vocab(train)
        TRG.build_vocab(train)

        #pickle.dump(SRC, open('weights/SRC.pkl', 'wb'))
        #pickle.dump(TRG, open('weights/TRG.pkl', 'wb'))
    else:
        try:
            with open('weights/SRC.pkl', 'rb') as pickle_file:
                SRC = pickle.load(pickle_file)
            with open('weights/TRG.pkl', 'rb') as pickle_file:
                TRG = pickle.load(pickle_file)

            print("SRC and TRG already exists, loaded.")
        except:
            print("SCR and TRG files not found")

    # Padding
    params['src_pad'] = SRC.vocab.stoi['<pad>']
    params['trg_pad'] = TRG.vocab.stoi['<pad>']

    train_iter = MyIterator(train, batch_size=params["batchsize"], device=params["device"],
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True, shuffle=True)

    params["train_len"] = get_len(train_iter)

    return train_iter
                                   
def get_len(train):

    for i, b in enumerate(train):
        pass
    
    return i                   