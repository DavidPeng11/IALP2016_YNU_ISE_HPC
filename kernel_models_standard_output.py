import gensim
import codecs
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn import svm



def load_data(type):
    if type == 'Word2Vec':
        model = gensim.models.Word2Vec.load("Word2Vec_model")

    elif type == 'Glove':
        model = gensim.models.Word2Vec.load_word2vec_format("GloVe_model", binary=False)

    model_list = list(model.vocab.keys())
    train_file = codecs.open("seed_words", 'r', 'utf8')
    seed_word_valence = dict()
    seed_word_arousal = dict()

    # The format of seed file is like this "words valence arousal"
    for line in train_file:
        line = line.strip().split('\t', -1)
        train_word = line[0]
        valence_line = line[1]
        arousal_line = line[2]
        seed_word_valence[train_word] = float(valence_line)
        seed_word_arousal[train_word] = float(arousal_line)



    test_file = codecs.open("test_words", 'r', 'utf8')
    test_word = []
    numbers = []
    for line in test_file:
        line = line.strip().split(',', -1)
        numbers.append(line[0].strip())
        test_word.append(line[1].strip())

    return model, model_list, seed_word_valence, seed_word_arousal, test_word, numbers


def make_matrix(model, word_1, word_2, value, kernel):
    x = []
    if kernel == 'linear':
        for i in word_1:
            x_row = [1]
            for j in word_2:
                d = model.similarity(i, j)
                f = d
                v = value[j]
                x_row.append(f * v)
            x.append(x_row)

    elif kernel == 'square':
        for i in word_1:
            x_row = [1]
            for j in word_2:
                d = model.similarity(i, j)
                f = d ** 2
                v = value[j]
                x_row.append(f * v)
            x.append(x_row)

    elif kernel == 'cube':
        for i in word_1:
            x_row = [1]
            for j in word_2:
                d = model.similarity(i, j)
                f = d ** 3
                v = value[j]
                x_row.append(f * v)
            x.append(x_row)

    x = np.array(x)
    return x


if __name__ == '__main__':
    # =========================================
    type = 'Glove'
    kernel = 'square'
    option = 'linear'
    # =========================================
    model, model_list, seed_word_valence, seed_word_arousal, test_word, numbers = load_data(type)
    train_words = filter(lambda a: a in model_list, seed_word_arousal.keys())

    y_valence = []
    y_arousal = []
    for i in train_words:
        y_valence.append(seed_word_valence[i])
        y_arousal.append(seed_word_arousal[i])
    y_valence = np.array(y_valence)

    y_arousal = np.array(y_arousal)

    x_train_valence = make_matrix(model, train_words, train_words, seed_word_valence, kernel)

    x_train_arousal = make_matrix(model, train_words, train_words, seed_word_arousal, kernel)

    x_test_valence = make_matrix(model, test_word, train_words, seed_word_valence, kernel)

    x_test_arousal = make_matrix(model, test_word, train_words, seed_word_arousal, kernel)


    if option == 'linear':
        lr = LinearRegression()
        lr.fit(x_train_valence, y_valence)
        y_valence_predict = lr.predict(x_test_valence)
        lr.fit(x_train_arousal, y_arousal)
        y_arousal_predict = lr.predict(x_test_arousal)
        output_file = codecs.open("output_file", 'w', 'utf8')
        for i in range(0, len(y_arousal_predict)):
            output_file.write(str(numbers[i]) + ', ' + str(round(y_valence_predict[i], 2)) + ', ' + str(
                round(y_arousal_predict[i], 2)) + '\n')
        output_file.close()


    elif option == 'SVR':
        svr_lin = svm.SVR(kernel='linear', C=1.0, epsilon=0.3)
        svr_lin.fit(x_train_valence, y_valence)
        y_valence_predict = svr_lin.predict(x_test_valence)
        svr_lin.fit(x_train_arousal, y_arousal)
        y_arousal_predict = svr_lin.predict(x_test_arousal)
        output_file = codecs.open("output_file", 'w', 'utf8')
        for i in range(0, len(y_arousal_predict)):
            output_file.write(str(numbers[i]) + ', ' + str(round(y_valence_predict[i], 2)) + ', ' + str(
                round(y_arousal_predict[i], 2)) + '\n')
        output_file.close()

