import sys
sys.path.append('/home/ywu10/Documents/MoralCausality/')
import torch
from dataloader.analysis import Analysis
import argparse
import spacy
import pandas as pd
from model.actorcritic import Critic,CalReward
from loguru import logger
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from dataloader.preprogress import WordDataset
from torchtext.legacy import data, datasets
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model.encoder import FastText, Bert, LSTMTagger
from dataloader.preprogress import BertDataLoader
from model.classifier import Classifier
import pickle
from transformers import BertConfig, BertForSequenceClassification, AdamW

parser = argparse.ArgumentParser(
    description='PyTorch TruncatedLoss')

parser.add_argument('--device', default=0)
parser.add_argument('--dataset', default='morality')
parser.add_argument('--n_class', default=11, type = int)
parser.add_argument('--N_GRAMS', default=1, type=int)
parser.add_argument('--MAX_LENGTH', default=None, type=int)
parser.add_argument('--VOCAB_MAX_SIZE', default=None, type=int)
parser.add_argument('--BATCH_SIZE', '-b', default=16,type=int)
parser.add_argument('--N_EPOCHS', default=30, type=int)
parser.add_argument('--START_TUNE', default=100, type=int)
parser.add_argument('--HIDDEN_DIM', default=300, type=int)
parser.add_argument('--OUTPUT_DIM', default=11, type=int)
parser.add_argument('--VOCAB_MIN_FREQ', default=1, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--model', default='lstm', type=str)
parser.add_argument('--using_glove', default=True)
parser.add_argument('--using_rl', default=False)
parser.add_argument('--stop_domain', default=180)
parser.add_argument('--source', default='BLM')
parser.add_argument('--target', default='ALM')
parser.add_argument('--train', default='train_rl')
parser.add_argument('--reward1', default=1.)
parser.add_argument('--reward2', default=1.)
args = parser.parse_args()


def generate_n_grams(x):

    N_GRAMS = 1
    if args.N_GRAMS <= 1: #no need to create n-grams if we only want uni-grams
        return x
    else:
        n_grams = set(zip(*[x[i:] for i in range(N_GRAMS)]))
        for n_gram in n_grams:
            x.append(' '.join(n_gram))
        return x

def func():
    dist, label, source = Analysis().distribution()
    if args.model == 'bert':
        from experiment.trainbert import train_b, train_rl, train_dann, train_mcd, train_jumbot,train_alda
        train_dataset = BertDataLoader(source,label,dist,args=args,source_data=args.source,target_data=args.target,train=True)
        test_dataset = BertDataLoader(source,label,dist,args=args,source_data=args.source,target_data=args.target,train=False)

        traindata = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=args.BATCH_SIZE, shuffle=True, num_workers=10)
        testdata = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=args.BATCH_SIZE, shuffle=False, num_workers=10)
        config = BertConfig.from_pretrained('bert-base-uncased')
        config.num_labels = 11
        encoder = Bert(config,args)
        embedding_size = 768

    else:
        from experiment.train import train_b, train_rl, train_dann, train_mcd, train_jumbot,train_alda
        TOKENIZER = lambda s: s.split( )
        TEXT = data.Field(batch_first=True, tokenize=TOKENIZER, preprocessing=generate_n_grams, fix_length=args.MAX_LENGTH)
        LABEL = data.Field(batch_first=True, use_vocab = False,sequential=True)

        if args.dataset == 'amazon' or args.dataset == 'news':
            LABEL = data.Field(sequential=False, use_vocab=False)
        train = WordDataset(args,TEXT,LABEL,source,label,source_area=args.source,target_area=args.target,train=True)
        test = WordDataset(args,TEXT,LABEL,source,label,source_area=args.source,target_area=args.target,train=False)

        LABEL.build_vocab(train,test)
        TEXT.build_vocab(train,test, max_size=args.VOCAB_MAX_SIZE, min_freq=args.VOCAB_MIN_FREQ,vectors="glove.6B.300d")

        traindata,testdata = data.BucketIterator.splits(
            (train, test),
            batch_size=args.BATCH_SIZE,
            sort_key=lambda x: max(len(x.review_s),len(x.review_t)),
            device = None if torch.cuda.is_available() else -1,
            repeat=False)

        if args.model == 'lstm':

            encoder = LSTMTagger(TEXT.vocab,args)
            embedding_size = 512

        elif args.model == 'fasttext':
            encoder = FastText(len(TEXT.vocab), args)
            embedding_size = args.HIDDEN_DIM

    encoder = encoder.cuda(args.device)
    classifier = Classifier(embedding_size,args).cuda(args.device)

    optimizer1 = optim.Adam(encoder.parameters(), args.lr)
    optimizer2 = optim.Adam(classifier.parameters(), args.lr)
    F = eval(args.train)(encoder,classifier,traindata,testdata, optimizer1,optimizer2,args)

    return F

if __name__ == '__main__':

    F = func()
    print(F)
