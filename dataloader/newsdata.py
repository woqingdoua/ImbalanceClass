import torchtext
import os
import pandas as pd
import random
from sklearn.utils import shuffle
#torchtext.datasets.IMDB(root='/home/ywu10/Documents/MoralCausality/data', split=('train', 'test'))
torchtext.datasets.YelpReviewPolarity(root='/home/ywu10/Documents/MoralCausality/data', split=('train', 'test'))

def IMDB():

    ratio = random.random()
    print(ratio)
    root = '/home/ywu10/Documents/MoralCausality/data/IMDB/aclImdb/train/neg'
    data_neg = read_data(root,labels=0)

    root = '/home/ywu10/Documents/MoralCausality/data/IMDB/aclImdb/train/pos'
    data_pos = read_data(root,labels=1)

    pos_ratio = int(ratio * 2000)
    neg_ratio = 2000 - pos_ratio

    data = shuffle(pd.concat([data_pos[:pos_ratio],data_neg[:neg_ratio]],axis=0))
    data['source'] = 'imdb'
    return data


def read_data(root,labels):
    label = []
    text = []
    for _,_,files in os.walk(root):
        label += [labels] * len(files)
        for file in files:
            f = os.path.join(root,file)
            f = open(f)
            lines = f.read()
            text.append([lines])
    data = pd.DataFrame({'review':text, 'label':label})
    return data

def YelpReviewPolarity():
    f = '/home/ywu10/Documents/MoralCausality/data/YelpReviewPolarity/yelp_review_polarity_csv/train.csv'
    a = pd.read_csv(f,header=0,names=['label','review'])
    a['label'] = a['label'] - 1
    ratio = random.random()
    print(ratio)
    pos_ratio = int(ratio*2000)
    pos = a[a['label'] == 1][:pos_ratio]
    neg_ratio = 2000 - pos_ratio
    neg = a[a['label'] == 0][:neg_ratio]
    data = shuffle(pd.concat([pos,neg],axis=0))
    data['source'] = 'yelp'

    return data

def news_preprogress():
    yelp = YelpReviewPolarity()
    imdb = IMDB()
    data = pd.concat([yelp,imdb],axis=0)
    data['tid'] = 0
    file_ = '/home/ywu10/Documents/MoralCausality/data/yelp_imdb_pregrocess.tsv'
    data.to_csv(file_, index = False)
