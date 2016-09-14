import gensim
import codecs
import numpy as np
import math
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr







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
                f = d**2
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
    elif kernel == 'exp':
        for i in word_1:
            x_row=[1]
            for j in word_2:
                d= model.similarity(i,j)
                f=math.exp(d)
                v= value[j]
                x_row.append(f*v)
            x.append(x_row)

    x = np.array(x)
    return x



if __name__ == '__main__':
    # ================================
    kernel='exp'
    # =================================
    # model = gensim.models.Word2Vec.load_word2vec_format("Glove.vectors.txt", binary=False)
    model = gensim.models.Word2Vec.load("wiki.zh.text.model")
    print 'model loaded'

    model_list = list(model.vocab.keys())

    seed_file = codecs.open("seed_sim.txt", 'r', 'utf8')

    print ('seed file loaded')

    seed_word_valence = dict()
    seed_word_arousal = dict()
    for line in seed_file:
        line = line.strip().split('\t', -1)
        word_line = line[0]
        valence_line = line[1]
        arousal_line = line[2]
        seed_word_valence[word_line] = float(valence_line)
        seed_word_arousal[word_line] = float(arousal_line)

    match_list = filter(lambda a: a in model_list, seed_word_arousal.keys())

    X=np.array(match_list)
    Y_valence=[]
    Y_arousal=[]
    for i in match_list:
        Y_valence.append(seed_word_valence[i])
        Y_arousal.append(seed_word_arousal[i])
    Y_valence=np.array(Y_valence)
    Y_arousal=np.array(Y_arousal)

    # output_file = codecs.open("C:\Users\Peng\Desktop\paper\kernel_l_v.txt", 'w', 'utf8')

    kf = KFold(len(Y_valence), n_folds=5, shuffle=False, random_state=None)
    for train_index, test_index in kf:
        x_word_train, x_word_test = X[train_index], X[test_index]
        # y_valence_train, y_valence_test = Y_valence[train_index], Y_valence[test_index]
        y_arousal_train, y_arousal_test = Y_arousal[train_index], Y_arousal[test_index]

        # x_valence_train = make_matrix(model, x_word_train,x_word_train, seed_word_valence, kernel)
        # x_valence_test = make_matrix(model,x_word_test, x_word_train, seed_word_valence, kernel)

        x_arousal_train= make_matrix(model, x_word_train,x_word_train,seed_word_arousal, kernel)
        x_arousal_test = make_matrix(model, x_word_test, x_word_train, seed_word_arousal, kernel)



        # lr_arousal=LinearRegression()
        lr=LinearRegression()

        # lr.fit(x_valence_train,y_valence_train)
        lr.fit(x_arousal_train, y_arousal_train)

        # y_predict = lr.predict(x_valence_test)
        y_predict=lr.predict(x_arousal_test)

        # rmse = math.sqrt(mean_squared_error(y_valence_test, y_predict))
        # # rmse = math.sqrt(mean_squared_error(y_arousal_test, y_predict))

        # mae = mean_absolute_error(y_valence_test, y_predict)
        mae = mean_absolute_error(y_arousal_test, y_predict)

        # r = pearsonr(y_valence_test, y_predict)[0]
        r = pearsonr(y_arousal_test, y_predict)[0]

        print mae, r



#

#
