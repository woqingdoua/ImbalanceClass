import torch
from loguru import logger
from experiment.metric import F1
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from copy import deepcopy
import random
import sklearn
import scipy
import numpy as np
from model.rl import RL
import pickle
import ot
from model.classifier import Classifier
from model.actorcritic import Critic,CalReward
from model.ALDAloss import ALDA_loss
from model.classifier import Discriminator
import torch.autograd as autograd

def train_b(encoder,classifier,traindata,testdata,optimizer1,optimizer2,args):

    best = 0
    for epoch in range(1, args.N_EPOCHS+1):

        encoder.train()
        classifier.train()
        for idx, batch in enumerate(traindata):

            x = batch.review_s.cuda(args.device)
            y = batch.label_s.float().cuda(args.device)
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            embedding,_ = encoder(x)
            prediction = classifier(embedding)
            loss = F.binary_cross_entropy(prediction,y)
            loss.backward()
            optimizer1.step()
            optimizer2.step()


        precision, recall, f1 = eval(encoder,classifier,testdata,epoch,args)

        if best < f1:
            best = f1
            a = precision, recall, f1

    return a[-1]

def train_dann(encoder,classifier,traindata,testdata,optimizer1,optimizer2,args):

    best = 0
    converage = []
    for epoch in range(1, args.N_EPOCHS+1):

        encoder.train()
        classifier.train()
        for idx, batch in enumerate(traindata):

            x_s = batch.review_s
            y_s = batch.label_s.float()
            x_t = batch.review_t
            y_t = batch.label_t.float()

            p = torch.tensor(epoch/(args.N_EPOCHS+1)).cuda(args.device)
            x_s = x_s.cuda(args.device)
            embedding,domain = encoder((x_s),p)
            x_s = x_s.cpu()
            prediction = classifier(embedding)
            y_s = y_s.cuda(args.device)
            loss = F.binary_cross_entropy(prediction,y_s)
            y_s = y_s.cpu()

            source = [0] * len(y_s) + [1] * len(y_t)
            source = torch.tensor(source)

            pad = torch.zeros(len(x_s),max(len(x_s),len(x_t)) - min(len(x_s),len(x_t)))
            if len(x_s)>len(x_t):
                x_t = torch.cat([x_s,pad],dim=-1)
            else:
                x_s = torch.cat([x_t,pad],dim=-1)
            x = torch.cat([x_s,x_t],dim=0)
            y = torch.cat([y_s,y_t],dim=0)
            if args.n_class == 2:
                data = torch.cat([x,y.unsqueeze(-1),source.unsqueeze(-1)],dim=-1).numpy()
            else:
                data = torch.cat([x,y,source.unsqueeze(-1)],dim=-1).numpy()
            random.shuffle(data)
            data = torch.from_numpy(data).cuda(args.device)

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            embedding,domain = encoder(data[:,:-13],p)

            mask = (data[:,-1] == 0).float().cuda(args.device)
            loss1 = 0.1*F.binary_cross_entropy(domain[-1], mask)
            loss = loss1 + loss

            loss.backward()
            optimizer1.step()
            optimizer2.step()

        precision, recall, f1 = eval(encoder,classifier,testdata,epoch,args)
        converage.append(f1)
        if best < f1:
            best = f1
            a = precision, recall, f1
    return a[-1]

def train_rl(encoder,classifier,traindata,testdata,optimizer1,optimizer2,args):

    best = 0
    critic = Critic(args)
    critic = critic.cuda(args.device)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=0.01, betas=(0.9, 0.99), eps=0.0000001)
    rl = RL(hidden_dim=256).cuda(args.device)
    optimizer_rl = torch.optim.Adam(rl.parameters(), lr=0.01, betas=(0.9, 0.99), eps=0.0000001)
    discriminator = Discriminator(classifier.hidden_dim).cuda(args.device)
    optimizer_d = optim.Adam(discriminator.parameters(), 0.01)
    calreward = CalReward(discriminator,classifier,args)

    for epoch in range(1, args.N_EPOCHS + 1):

        encoder.train()
        classifier.train()
        for batch in traindata:

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            x_s = batch.review_s.cuda(args.device)
            y_s = batch.label_s.float().cuda(args.device)

            embedding,_ = encoder(x_s)
            prediction = classifier(embedding)
            loss = F.binary_cross_entropy(prediction,y_s)
            loss.backward()
            optimizer1.step()
            optimizer2.step()

            #training discriminator
            optimizer_d.zero_grad()
            x_s = batch.review_s.cuda(args.device)
            x_t = batch.review_t.cuda(args.device)
            embedding_s,_ = encoder(x_s)
            pro_s = discriminator(embedding_s)
            pro_s_ = torch.tensor(len(x_s)*[1.]).cuda(args.device)
            loss = F.binary_cross_entropy(pro_s,pro_s_)
            embedding_t,_ = encoder(x_t)
            pro_t = discriminator(embedding_t)
            pro_t_ = torch.tensor(len(x_t)*[0.]).cuda(args.device)
            loss = F.binary_cross_entropy(pro_t,pro_t_) + loss
            loss.backward()
            optimizer_d.step()

            optimizer_rl.zero_grad()
            optimizer_critic.zero_grad()
            x_t = batch.review_t.cuda(args.device)
            x_s = batch.review_s.cuda(args.device)
            embedding_s,_ = encoder(x_s)
            embedding_t,_ = encoder(x_t)
            embedding = torch.cat((embedding_s,embedding_t),dim=0)

            embedding_m,log_softmax, entropy,out,action = rl(embedding)
            y_s = torch.tensor(len(embedding_s)*[1.]).cuda(args.device)
            y_m = torch.tensor(len(embedding_t)*[0.]).cuda(args.device)
            source = torch.cat((y_s,y_m),dim=0)
            value = calreward.reward(embedding,embedding_m,source,action)
            prediction = critic(embedding_m)
            loss1 = F.mse_loss(prediction.squeeze(),value)
            loss1.backward(retain_graph=True)
            optimizer_critic.step()
            prediction = critic(embedding_m)
            loss2 = rl.cal_loss(value,prediction,log_softmax, entropy)
            loss2.backward()
            optimizer_rl.step()

        encoder.eval()
        classifier.eval()
        discriminator.eval()
        prediction_list = []
        target_list = []
        for idx, batch in enumerate(testdata):

            target_list += batch.label_t.tolist()
            x = batch.review_t.cuda(args.device)

            embedding,_ = encoder(x)
            embedding = rl(embedding)[0]
            prediction = classifier(embedding)

            prediction_list += prediction.cpu().tolist()
        precision, recall, f1 = F1(prediction_list,target_list,epoch)

        if best < f1:
            best = f1
            a = precision, recall, f1

    return a[-1]

def train_mcd(extractor,classifier1,traindata,testdata,opt_e,opt_c1,args):

    '''
    Maximum Classifier Discrepancy for Unsupervised Domain Adaptation
    '''
    best = 0
    classifier2 = Classifier(classifier1.hidden_dim,args).cuda(args.device)
    opt_c2 = torch.optim.Adam(classifier2.parameters(), lr=args.lr)
    for epoch in range(1, args.N_EPOCHS+1):

        extractor.train()
        classifier1.train()
        classifier2.train()

        for idx, batch in enumerate(traindata):

            x = batch.review_s.cuda(args.device)
            y = batch.label_s.float().cuda(args.device)

            '''
            STEP A
            '''
            opt_e.zero_grad()
            opt_c1.zero_grad()
            opt_c2.zero_grad()

            src_feat, domain = extractor(x)
            preds_s1 = classifier1(src_feat)
            preds_s2 = classifier2(src_feat)

            loss_A =  F.binary_cross_entropy(preds_s1, y) + F.binary_cross_entropy(preds_s2, y)
            loss_A.backward()

            opt_e.step()
            opt_c1.step()
            opt_c2.step()

            '''
            STEP B
            '''
            x2 = batch.review_t.cuda(args.device)

            opt_e.zero_grad()
            opt_c1.zero_grad()
            opt_c2.zero_grad()

            src_feat,domain = extractor(x)
            preds_s1 = classifier1(src_feat)
            preds_s2 = classifier2(src_feat)
            loss_B = F.binary_cross_entropy(preds_s1, y) + F.binary_cross_entropy(preds_s2, y)

            src_feat,domain = extractor(x2)
            preds_t1 = classifier1(src_feat)
            preds_t2 = classifier2(src_feat)

            loss_B  = loss_B - torch.mean(torch.abs(preds_t1 - preds_t2))
            loss_B.backward()

            opt_c1.step()
            opt_c2.step()

            opt_e.zero_grad()
            opt_c1.zero_grad()
            opt_c2.zero_grad()

            '''
            STEP C
            '''
            N = 1
            for i in range(N):
                feat_tgt,domain = extractor(x2)
                preds_t1 = classifier1(feat_tgt)
                preds_t2 = classifier1(feat_tgt)
                loss_C = torch.mean(torch.abs(preds_t1- preds_t2))
                loss_C.backward()
                opt_e.step()

                opt_e.zero_grad()
                opt_c1.zero_grad()
                opt_c2.zero_grad()

        precision, recall, f11 = eval(extractor,classifier1,testdata,epoch,args)
        if best < f11:
            best = f11
            a = precision, recall, f11
        precision, recall, f12 = eval(extractor,classifier2,testdata,epoch,args)
        if best < f12:
            best = f12
            a = precision, recall, f12

    extractor.eval()
    x_s_mean = torch.zeros(512).cuda(args.device)
    for idx, batch in enumerate(traindata):
        x_s = batch.review_s.cuda(args.device)
        g_xs,_ = extractor(x_s)
        x_s_mean = torch.mean(g_xs,dim=0) + x_s_mean
    x_s_mean = x_s_mean/len(traindata)

    x_t_mean = torch.zeros(512).cuda(args.device)
    for idx, batch in enumerate(testdata):
        x_t = batch.review_t.cuda(args.device)
        g_xt,_ = extractor(x_t)
        x_t_mean = torch.mean(g_xt,dim=0) + x_t_mean
    x_t_mean = x_t_mean/len(testdata)
    dis = (x_s_mean - x_t_mean)**2
    print(torch.sum(dis))
    return a[-1]

def train_jumbot(encoder,classifier,traindata,testdata,optimizer1,optimizer2,args):
    #https://github.com/kilianFatras/JUMBOTe
    best = 0
    alpha = 0.01
    lambda_t = 0.5
    reg_m = 0.5

    for epoch in range(1, args.N_EPOCHS+1):

        encoder.train()
        classifier.train()
        for idx, batch in enumerate(traindata):

            x_s = batch.review_s.cuda(args.device)
            y_s = batch.label_s.float().cuda(args.device)
            x_t = batch.review_t.cuda(args.device)

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            g_xs, domain_s = encoder(x_s)
            pred_s = classifier(g_xs)

            g_xt, domain_t = encoder(x_t)
            pred_t = classifier(g_xt)

            loss_A =  F.binary_cross_entropy(pred_s, y_s)

            M_embed = torch.cdist(g_xs, g_xt)**2
            try:
                M_sce = - torch.mm(y_s, torch.transpose(torch.log(pred_t), 0, 1))  # Term on labels
            except:
                M_sce = - torch.mm(y_s.unsqueeze(-1), torch.transpose(torch.log(pred_t).unsqueeze(-1), 0, 1))
            M = alpha * M_embed + lambda_t * M_sce

            #OT computation
            a, b = ot.unif(g_xs.size()[0]), ot.unif(g_xt.size()[0])
            pi = ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, M.detach().cpu().numpy(),0.01, reg_m=reg_m)
            pi = torch.from_numpy(pi).float().cuda(args.device)  # Transport plan between minibatches
            transfer_loss = torch.sum(pi * M)

            total_loss = loss_A + transfer_loss
            total_loss.backward()
            optimizer1.step()
            optimizer2.step()

        precision, recall, f1 = eval(encoder,classifier,testdata,epoch,args)
        if best < f1:
            best = f1
            aa = precision, recall, f1
    return aa[-1]

def eval(encoder,classifier,testdata,epoch,args):

    encoder.eval()
    classifier.eval()
    prediction_list = []
    target_list = []
    for idx, batch in enumerate(testdata):

        target_list += batch.label_t.tolist()
        x = batch.review_t.cuda(args.device)

        embedding,_ = encoder(x)
        prediction = classifier(embedding)

        prediction_list += prediction.cpu().tolist()

    precision, recall, f1 = F1(prediction_list,target_list,epoch)

    return precision, recall, f1

def train_alda(encoder,classifier,traindata,testdata,optimizer1,optimizer2,args):
    #https://github.com/ZJULearning/ALDA/tree/09df8864fde9d25d25ad22e8c62f135d3d7596d5
    classifier2 = Classifier(classifier.hidden_dim,args).cuda(args.device)
    opt_c2 = torch.optim.Adam(classifier2.parameters(), lr=args.lr)

    best = 0
    for epoch in range(1, args.N_EPOCHS+1):

        encoder.train()
        classifier.train()
        classifier2.train()

        for idx, batch in enumerate(traindata):

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            opt_c2.zero_grad()

            inputs_source = batch.review_s.cuda(args.device)
            y1 = batch.label_s.float().cuda(args.device)
            inputs_target = batch.review_t.cuda(args.device)

            features_source, domain_s = encoder(inputs_source)
            outputs_source = classifier(features_source)

            features_target, domain_s = encoder(inputs_target)
            outputs_target = classifier(features_target)

            features = torch.cat((features_source, features_target), dim=0)
            outputs = torch.cat((outputs_source, outputs_target), dim=0)

            ad_out = classifier2(features)

            adv_loss, reg_loss, correct_loss = ALDA_loss(ad_out, y1, outputs,args)

            # whether add the corrected self-training loss
            trade_off = np.float(2.0/ (1.0 + np.exp(-10* epoch / 10000)) - 1)
            transfer_loss = adv_loss + trade_off * correct_loss

            for param in encoder.parameters():
                param.requires_grad = False
            reg_loss.backward(retain_graph=True)
            for param in encoder.parameters():
                param.requires_grad = True

            classifier_loss = F.binary_cross_entropy(outputs_source, y1)
            total_loss = classifier_loss + transfer_loss
            total_loss.backward()
            optimizer1.step()
            optimizer2.step()
            opt_c2.step()

        precision, recall, f1 = eval(encoder,classifier,testdata,epoch,args)

        if best < f1:
            best = f1
            a = precision, recall, f1

    return a[-1]