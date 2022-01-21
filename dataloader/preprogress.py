import json
import numpy as np
import pandas as pd
import torch
import pickle
import os
import torchtext as text
from torchtext import data
import torch.nn.functional as F
from torchtext.legacy import data, datasets
from torchtext.vocab import Vectors
from transformers import BertTokenizer
from sklearn.utils import shuffle
from numpy import random
from sklearn.utils import shuffle
from scipy.special import kl_div
from dataloader.newsdata import news_preprogress

def amazonprogress():
    data_ = ['book','dvd','kitchen', 'eletronic']
    file_ = '/home/ywu10/Documents/MoralCausality/data/amazon_preprogress.tsv',
    files = ['/home/ywu10/Documents/MoralCausality/data/Books.json', \
             '/home/ywu10/Documents/MoralCausality/data/Electronics.json',
             '/home/ywu10/Documents/MoralCausality/data/CDs_and_Vinyl.json', \
             '/home/ywu10/Documents/MoralCausality/data/Home_and_Kitchen.json', \
             ]
    tid = []
    text = []
    source = []
    labels = []
    for file in range(len(files)):
        neg_num = int(2000*random.random())
        pos_num = 2000 - neg_num
        print(neg_num)

        with open(files[file], 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            if neg_num>0 or pos_num>0:
                dd = json.loads(line)
                try:
                    review = dd['reviewText']
                    if dd['verified'] == True and dd['overall'] != 3 and len(review)<500:
                        score = dd['overall']
                        if score>3 and pos_num >0 :
                            tid.append(dd['reviewerID'])
                            text.append(review)
                            source.append(data_[file])
                            labels.append(1)
                            pos_num -= 1

                        elif score<3 and neg_num > 0 :
                            tid.append(dd['reviewerID'])
                            text.append(review)
                            source.append(data_[file])
                            labels.append(0)
                            neg_num -= 1
                except:
                    continue
            else:
                break
    data = pd.DataFrame({'tid':tid, 'text':text, 'source':source, 'label':labels})
    data.to_csv(file_, index = False)


def preprogress():
    file_address = '/data/datasets/moralcausality/MFTC_V4_text.json'
    file_ = '/home/ywu10/Documents/MoralCausality/data/MFTC_V4_text_preprogress.tsv'
    f = open(file_address)
    data = json.load(f)
    tid = []
    text = []
    source = []
    labels = []
    for corpus in data:
        for id in corpus['Tweets']:
            label = []
            for ll in id['annotations']:
                label += ll['annotation'].split(',')
            rep = list(set(label))
            label_vote = []
            for ll in rep:
                if (np.array(label) == ll).sum() > 1:
                    label_vote.append(ll)
            label_vote = ','.join(label_vote)
            if len(label_vote) > 0:
                source.append(corpus['Corpus'])
                tid.append(id['tweet_id'])
                text.append(id['tweet_text'])
                labels.append(label_vote)
    data = pd.DataFrame({'tid':tid, 'text':text, 'source':source, 'label':labels})
    data.to_csv(file_, index = False)

class WordDataset(data.Dataset):
    def __init__(self,args,TEXT,LABEL, source,label, source_area, target_area, \
                 file='/home/ywu10/Documents/MoralCausality/data/MFTC_V4_text_preprogress.tsv', train=True,**kwargs):

        self.file = file
        self.train = train
        self.args = args
        self.TEXT = TEXT
        self.LABEL = LABEL
        self.source = source
        self.label = label
        self.source_area = source_area
        self.target_area = target_area
        self.file_ = '/home/ywu10/Documents/MoralCausality/data/MFTC_V4_text_word.csv'
        if args.dataset == 'amazon':
            self.file = '/home/ywu10/Documents/MoralCausality/data/amazon_preprogress.tsv'
            self.file_ = '/home/ywu10/Documents/MoralCausality/data/amazon_text_word.csv'
        if args.dataset == 'news':
            self.file_ = '/home/ywu10/Documents/MoralCausality/data/yelp_imdb_pregrocess.tsv'
        examples, fields = self.example()

        super(WordDataset, self).__init__(examples, fields, **kwargs)

    def coding(self):

        if os.path.exists(self.file_) == False:
            if os.path.exists(self.file) == False:
                if self.args.dataset =='amazon':
                    amazonprogress()
                elif self.args.dataset =='news':
                    news_preprogress()
                else:
                    preprogress()
            data = pd.read_csv(self.file)
            tid = list(set(data['tid']))
            tid_code = []
            for code in data['tid']:
                tid_code.append(tid.index(code))

            source_code = []
            for code in data['source']:
                source_code.append(code)

            reviews = data['text'].values.tolist() ##提取rating信息并保存为list格式

            train_data = pd.DataFrame({'tid':tid_code, 'review':reviews, 'source':source_code, 'label':data['label']})
            train_data.to_csv(self.file_, index = False)

    def example(self):

        examples = []

        fields = [('tid',None),('review_s',self.TEXT), ('label_s',self.LABEL), ('review_t',self.TEXT), ('label_t',self.LABEL)]

        self.coding()
        tsv_data = pd.read_csv(self.file_)

        if self.train:
            data_s = tsv_data[tsv_data['source'] == self.source_area]
            data_t = tsv_data[tsv_data['source'] == self.target_area]
            split = int(min(len(data_s),len(data_t)) * 0.8)
            data_s = data_s[:split]
            data_t = data_t[:split]

            pos_s = len(data_s[data_s['label'] == 1])/len(data_s)
            pos_t = len(data_t[data_t['label'] == 1])/len(data_t)
            kl1 = kl_div([pos_s,1-pos_s],[pos_t,1-pos_t])
            kl2 = kl_div([pos_t,1-pos_t],[pos_s,1-pos_s])
            #print(kl1,kl2)

        else:
            data_s = tsv_data[tsv_data['source'] == self.target_area]
            data_t = tsv_data[tsv_data['source'] == self.source_area]
            length = min(int(len(data_s) * 0.2), int(len(data_t) * 0.2))
            data_s = data_s[-length:-1]
            data_t = data_t[-length:-1]

        max_len = 0
        for text_s, label_s, text_t, label_t in zip(data_s['review'], data_s['label'], data_t['review'], data_t['label']):
            if self.args.dataset == 'morality':
                labels = 11*[0]
                label_rep = label_s.split(',')
                for la in label_rep:
                    labels[self.label.index(la)] = 1

                labelt = 11*[0]
                label_rep = label_s.split(',')
                for la in label_rep:
                    labelt[self.label.index(la)] = 1
            else:
                labelt = label_t
                labels = label_s

            text_s = text_s[:200]
            text_t = text_t[:200]
            a = max(len(text_s),len(text_t))
            if a > max_len:
                max_len = a
            examples.append(data.Example.fromlist([None, text_s, labels, text_t, labelt], fields))

        return examples,fields


class BertDataLoader(data.Dataset):
    def __init__(self,source, label,dist,source_data,target_data,args, file = '/home/ywu10/Documents/MoralCausality/data/MFTC_V4_text_preprogress.tsv', \
                 file__ = '/home/ywu10/Documents/MoralCausality/data/MFTC_V4_text_preprogress.pkl',train=True, max_len = None):

        self.source = source
        if args.dataset == 'amazon':
            file = '/home/ywu10/Documents/MoralCausality/data/amazon_preprogress.tsv'
            file__ = '/home/ywu10/Documents/MoralCausality/data/amazon_preprogress.pkl'

        elif args.dataset == 'news':
            file = '/home/ywu10/Documents/MoralCausality/data/yelp_imdb_pregrocess.tsv'
            file__ = '/home/ywu10/Documents/MoralCausality/data/yelp_imdb_preprogress.pkl'

        if os.path.exists(file__) == False:
            data = pd.read_csv(file)
            tid = list(set(data['tid']))
            tid_code = []
            for code in data['tid']:
                tid_code.append(tid.index(code))

            source_code = []
            for code in data['source']:
                source_code.append(code)

            if args.dataset == 'morality':
                label_code = []
                for code in data['label']:
                    label_ = np.zeros(len(label))
                    label_rep = code.split(',')
                    for la in label_rep:
                        label_[label.index(la)] = 1
                    label_code.append(label_)
            else:
                label_code = data['label']

            if max_len == None:
                #len_list = [len(i) for i in data['text']]
                max_len = 200#int(np.percentile(len_list, 85))

            try:
                reviews = data['text'].values.tolist() ##提取rating信息并保存为list格式
            except:
                reviews = data['review'].values.tolist()
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            review_ = []
            mask_ = []
            for i in range(len(reviews)):
                review = reviews[i]
                if len(reviews[i])>max_len:
                    review = review[:max_len]
                dd = tokenizer(review,return_tensors="pt")
                pad = torch.tensor(int((max_len - len(dd['input_ids'].squeeze()))) * [0])
                data = torch.cat((dd['input_ids'].squeeze(),pad), dim=0)
                mask = torch.cat((dd['attention_mask'].squeeze(),pad), dim=0)
                review_.append(data)
                mask_.append(mask)

            data = tid_code, source_code, label_code, review_, mask_
            with open(file__, 'wb') as fo:
                pickle.dump(data, fo)

        with open(file__, 'rb') as fo:
            data = pickle.load(fo)

        self.train = train
        self.dist = dist
        self.source = source
        tsv_data = pd.DataFrame({'tid':data[0], 'review':data[3],'mask':data[4], 'source':data[1], 'label':data[2]})

        sdata = tsv_data[tsv_data['source'] == source_data]
        tdata = tsv_data[tsv_data['source'] == target_data]
        length = min(len(sdata),len(tdata))
        if self.train:
            split = int(length* 0.8)
            sdata = sdata[:split]
            tdata = tdata[:split]

        else:
            split = int(length * 0.2)
            sdata = sdata[-split:]
            tdata = tdata[-split:]

        self.datas = sdata['tid'].tolist(),sdata['label'].tolist(),sdata['review'].tolist(),sdata['mask'].tolist(), \
                     tdata['tid'].tolist(),tdata['label'].tolist(),tdata['review'].tolist(),tdata['mask'].tolist()

    def __getitem__(self, index):

        s = self.datas[0][index], self.datas[1][index], self.datas[2][index], self.datas[3][index]
        t = self.datas[4][index], self.datas[5][index], self.datas[6][index], self.datas[7][index]
        return s,t

    def __len__(self):
        return len(self.datas[0])
