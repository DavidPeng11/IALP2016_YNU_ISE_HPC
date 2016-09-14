import gensim
import codecs
import math
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr



def test(length, x, y, output):
    kf = KFold(length, n_folds=5, shuffle=False, random_state=None)
    for train_index, test_index in kf:
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        lr= LinearRegression()
        lr.fit(x_train, y_train)
        y_lin = lr.predict(x_test)
        rmse = math.sqrt(mean_squared_error(y_test, y_lin))
        mae = mean_absolute_error(y_test, y_lin)
        r = pearsonr(y_test, y_lin)[0]
        output.write(str(rmse) + '\t' + str(mae) + '\t' + str(r) + '\n')

if __name__ == '__main__':

    # model = gensim.models.Word2Vec.load("C:\Pycharm\Projects\Word2Vec\New\wiki.zh.text.add.model")
    model = gensim.models.Word2Vec.load_word2vec_format("Glove.vectors.txt", binary=False)
    print ('model loaded')

    seed_file = codecs.open("seed_sim.txt", 'r', 'utf8')
    print ('seed file loaded')

    model_list = list(model.vocab.keys())

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

    X = []
    for i in match_list:
        X.append(model[i])

    X = np.array(X)

    y_valence = []
    y_arousal = []
    for i in match_list:
        y_valence.append(seed_word_valence[i])
        y_arousal.append(seed_word_arousal[i])
    y_arousal = np.array(y_arousal)
    y_valence = np.array(y_valence)

    output_file = codecs.open('C:\Users\Peng\Desktop\paper\VR_G.text', 'w', 'utf8')
    output_file.write('RMSE\tMAE\tPearsonr\n')

    output_file.write('Valence\n')
    test(len(y_valence), X, y_valence, output_file)
    output_file.write('Arousal\n')
    test(len(y_arousal), X, y_arousal, output_file)
    output_file.close()


