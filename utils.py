### COPIED FILE FROM HyCorpus ###
### DON'T use !               ###
### use bytetokenizer instead ###

import numpy as np
from tokenizers import pre_tokenizers, normalizers, Regex, PreTokenizedString





def save_vocab(vocab, vocab_path):
    with open(vocab_path+'/vocab.txt', mode='w') as file:
        file.write(vocab)

def load_vocab(vocab_path):
    with open(vocab_path+'/vocab.txt', mode='r') as file:
        vocab = file.read()
    return vocab

class CharTokenizer():
    def __init__(self, pad_token, unk_token):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.normalizer = normalizers.Replace(Regex(r'(?=[\t\n])|(?<=[\t\n])'), ' ')
        self.pre_tokenizer = pre_tokenizers.Split(Regex(r' +'), behavior='removed')
    
    def train(self, filepath):
        # define later
        pass

    def update_vocab(self):
        self.char_set = set(self.chars)

        self.char2id = {'[PAD]': 0, '[UNK]': 1}
        i = 2
        for c in self.chars:
            self.char2id[c] = i
            i += 1
        
        self.id2char = {}
        for c in self.char2id.keys():
            self.id2char[self.char2id[c]] = c
    
    def encode_1d(self, sequence, max_len=None):
        encoded = [self.char2id[x] if x in self.char_set else self.char2id['[UNK]'] for x in sequence]
        if max_len:
            encoded = encoded[:max_len]
            encoded = encoded + [self.pad_token]*(max_len - len(sequence))
        return encoded

    def encode_2d(self, sequences, max_len_w, max_len):
        encoded = [[self.char2id[x] if x in self.char_set else self.char2id['[UNK]'] for x in sequence][:max_len_w] + [self.pad_token]*max(max_len_w - len(sequence), 0)\
                   for sequence in sequences[:max_len]]
        return encoded + [[self.pad_token] * max_len_w ]* max(max_len - len(sequences), 0)
    
    def split_words(self, text):
        words = self.pre_tokenizer.pre_tokenize_str(self.normalizer.normalize_str(text))
        return [word[0] for word in words]
    
    def encode_2d_batch(self, batch, max_len, max_len_w):
        batch = [self.split_words(text) for text in batch]
        batch_max_len = max([len(words) for words in batch])
        encoded = [self.encode_2d(words, max_len_w=max_len_w, max_len=min(max_len, batch_max_len)) for words in batch]
        return encoded
    
    def token_to_id(self, token):
        if token in self.char_set:
            return self.char2id[token]
        else:
            return self.unk_token
    
    def id_to_token(self, id):
        return self.id2char[id]
    
    def decode(self, ids):
        decoed_chars = [self.id_to_token(id) for id in ids]
        return ''.join(decoed_chars)

    def from_file(self, filepath):
        self.chars = load_vocab(filepath)
        self.update_vocab()
        return self
    
    def get_vocab_size(self):
        return len(self.char2id)
