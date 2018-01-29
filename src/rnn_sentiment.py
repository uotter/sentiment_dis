from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
import keras.preprocessing.sequence as S
from keras.utils import to_categorical
from keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense

import jieba
import json
import numpy as np

vocab_size = 350000
sentence_max_len = 100
model_path = 'keras.h5'

class SentimentLSTM:
    def __init__(self):
        self.tokenizer = Tokenizer(num_words=vocab_size)
        self.stop_words = []
        self.model = None

    def load_stop_word(self,path='dict/stop_word.txt'):
        with open(path, 'r') as f:
            for line in f:
                content = line.strip()
                self.stop_words.append(content.decode('utf-8'))

    def jieba_cut(self,line):
        lcut = jieba.lcut(line)
        cut = [x for x in lcut if x not in self.stop_words]
        cut = " ".join(cut)
        return cut

    def load_cuted_corpus(self, dir, input):
        f = open(dir + '/' + input , 'r')
        lines = f.readlines()
        texts = []
        labels = []
        for line in lines:
            fields = line.split()
            rate = int(fields[0])
            if rate==0 or rate==3:
                continue
            elif rate < 3:
                rate = 0
            else:
                rate = 1
            cont = fields[1:]
            cont = " ".join(cont)
            texts.append(cont)
            labels.append(rate)

        self.tokenizer.fit_on_texts(texts)
        f.close()
        return texts,labels

    def load_data(self):
        x,y = self.load_cuted_corpus('corpus', 'review.csv')
        x = self.tokenizer.texts_to_sequences(x)
        x = S.pad_sequences(x,maxlen=sentence_max_len)
        y = to_categorical(y,num_classes=2)
        return ((x[0:500000],y[0:500000]), (x[500000:], y[500000:]))

    def train(self,epochs=50):
        print('building model ...')
        self.model = SentimentLSTM.build_model()

        print('loading data ...')
        (text_train, rate_train), (text_test, rate_text) = self.load_data()

        print('training model ...')
        self.model.fit(text_train, rate_train,batch_size=1000,epochs=epochs)
        self.model.save('model/keras.model')
        score = self.model.evaluate(text_test,rate_text)
        print(score)

    def load_trained_model(self,path):
        model = SentimentLSTM.build_model()
        model.load_weights(path)
        return model

    def predict_text(self,text):
        if self.model == None:
            self.model = self.load_trained_model(model_path)
            self.load_stop_word()
            self.load_cuted_corpus('corpus', 'review.csv')

        vect = self.jieba_cut(text)
        vect = vect.encode('utf-8')
        vect = self.tokenizer.texts_to_sequences([vect,])
        print(vect)
        return self.model.predict_classes(S.pad_sequences(np.array(vect),100))

    @staticmethod
    def build_model():
        model = Sequential()
        model.add(Embedding(vocab_size, 256, input_length=sentence_max_len))
        model.add(Bidirectional(LSTM(128,implementation=2)))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='relu'))
        model.compile('RMSprop', 'categorical_crossentropy', metrics=['accuracy'])
        return model

def main():
    lstm = SentimentLSTM()
    lstm.train(50)
    while True:
        input_str = input('Please input text:')
        if input_str == 'quit':
            break
        print(lstm.predict_text(input_str))

if __name__=="__main__":
    main()