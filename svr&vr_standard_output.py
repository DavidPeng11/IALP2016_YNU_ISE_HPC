#coding:utf-8
import gensim
import codecs
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import svm



def vector_regression(x_train, y_valence_train, y_arousal_train, x_test, number):
    lr = LinearRegression()
    lr.fit(x_train, y_valence_train)
    y_valence_predict = lr.predict(x_test)
    lr.fit(x_train, y_arousal_train)
    y_arousal_predict = lr.predict(x_test)

    output_file = codecs.open("output_vr.txt", 'w', 'utf8')

    for i in range(0, len(y_valence_predict)):
        output_file.write(str(number[i]) + ', ' + str(round(y_valence_predict[i], 2)) + ', ' + str(
            round(y_arousal_predict[i], 2)) + '\n')
    output_file.close()

def sv_regression(x_train, y_valence_train, y_arousal_train, x_test, number):
    svr_lin = svm.SVR(kernel='linear', C=1.0, epsilon=0.2)
    svr_lin.fit(x_train, y_valence_train)
    y_valence_predict = svr_lin.predict(x_test)
    svr_lin.fit(x_train, y_arousal_train)
    y_arousal_predict = svr_lin.predict(x_test)

    output_file = codecs.open("output_svr.txt", 'w', 'utf8')

    for i in range(0, len(y_valence_predict)):
        output_file.write(str(number[i]) + ', ' + str(round(y_valence_predict[i], 2)) + ', ' + str(
            round(y_arousal_predict[i], 2)) + '\n')
    output_file.close()


def load_data(dtype):
    if dtype == 'Word2Vec':
        model = gensim.models.Word2Vec.load("wiki.zh.text.add.model")

    elif dtype == 'Glove':
        model = gensim.models.Word2Vec.load_word2vec_format("Glove.vectors.txt", binary=False)

    model_list = list(model.vocab.keys())
    train_file = codecs.open("seed_sim.txt", 'r', 'utf8')
    seed_word_valence = dict()
    seed_word_arousal = dict()
    for line in train_file:
        line = line.strip().split('\t', -1)
        train_word = line[0]
        valence_line = line[1]
        arousal_line = line[2]
        seed_word_valence[train_word] = float(valence_line)
        seed_word_arousal[train_word] = float(arousal_line)

    test_file = codecs.open("Test.txt", 'r', 'utf8')
    test_word = []
    numbers=[]
    for line in test_file:
        line = line.strip().split(',', -1)
        numbers.append(line[0].strip())
        test_word.append(line[1].strip())

    return model, model_list, seed_word_valence, seed_word_arousal, test_word, numbers


if __name__ == '__main__':
    dtype = 'Glove'
    model, model_list, seed_word_valence, seed_word_arousal, test_word, numbers = load_data(dtype)
    train_match_list = filter(lambda a: a in model_list, seed_word_valence.keys())

    train_vector = []
    for i in train_match_list:
        train_vector.append(model[i])
    train_vector = np.array(train_vector)

    test_vector = []
    for i in test_word:
        test_vector.append(model[i])
    test_vector = np.array(test_vector)

    y_valence = []
    y_arousal = []
    for i in train_match_list:
        y_valence.append(seed_word_valence[i])
        y_arousal.append(seed_word_arousal[i])
    y_valence = np.array(y_valence)
    y_arousal = np.array(y_arousal)
    sv_regression(train_vector, y_valence, y_arousal, test_vector , numbers)




