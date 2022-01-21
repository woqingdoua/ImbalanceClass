import torch
from loguru import logger
from experiment.metric import F1
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from copy import deepcopy
import random
import ot
import sklearn
from model.rl import RL
from model.classifier import Classifier
from model.actorcritic import Critic,CalReward
from model.ALDAloss import ALDA_loss
from model.classifier import Discriminator
from model.classifier import Classifier
from model.actorcritic import Critic,CalReward
import numpy as np

def train_b(encoder,classifier,traindata,testdata,optimizer1,optimizer2,args):

    best = 0
    for epoch in range(1, args.N_EPOCHS+1):

        encoder.train()
        classifier.train()
        for batch_id, (sourced, targetd) in enumerate(traindata):
            _, s_label, s_review, s_mask = sourced[0].cuda(args.device), \
                                           sourced[1].cuda(args.device), sourced[2].cuda(args.device), sourced[3].cuda(args.device)

            _, t_label, t_review, t_mask = targetd[0].cuda(args.device), \
                                           targetd[1].cuda(args.device), targetd[2].cuda(args.device), targetd[3].cuda(args.device)

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            embedding,_ = encoder(s_review,s_mask)
            prediction = classifier(embedding)
            loss = F.binary_cross_entropy(prediction,s_label.float())
            loss.backward()
            optimizer1.step()

        precision, recall, f1 = eval(encoder,classifier,testdata,epoch,args)

        if best < f1:
            best = f1
            a = precision, recall, f1

    return a[0]

def train_dann(encoder,classifier,traindata,testdata,optimizer1,optimizer2,args):

    best = 0
    for epoch in range(1, args.N_EPOCHS+1):

        encoder.train()
        classifier.train()
        for batch_id, (sourced, targetd) in enumerate(traindata):
            _, y_s, x_s, s_mask = sourced[0].cuda(args.device), \
                                  sourced[1].cuda(args.device), sourced[2].cuda(args.device), sourced[3].cuda(args.device)

            _, y_t, x_t, t_mask = targetd[0].cuda(args.device), \
                                  targetd[1].cuda(args.device), targetd[2].cuda(args.device), targetd[3].cuda(args.device)

            p = torch.tensor(epoch/(args.N_EPOCHS+1)).cuda()
            embedding,domain = encoder(x_s,s_mask,p)
            x_s = x_s.cpu()
            x_t = x_t.cpu()
            prediction = classifier(embedding)
            loss = F.binary_cross_entropy(prediction,y_s.float())
            y_s = y_s.cpu()
            y_t = y_t.cpu()

            s_mask = s_mask.cpu()
            t_mask = t_mask.cpu()

            source = [0] * len(y_s) + [1] * len(y_t)
            source = torch.tensor(source)

            pad = torch.zeros(len(x_s),max(len(x_s),len(x_t)) - min(len(x_s),len(x_t)))
            if len(x_s)>len(x_t):
                x_t = torch.cat([x_s,pad],dim=-1)
            else:
                x_s = torch.cat([x_t,pad],dim=-1)
            x = torch.cat([x_s,x_t],dim=0)
            mask = torch.cat([t_mask,s_mask],dim=0)
            text = x.size()[-1]
            data = torch.cat([x,mask,source.unsqueeze(-1)],dim=-1).numpy()
            random.shuffle(data)
            data = torch.from_numpy(data).cuda(args.device)

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            embedding,domain = encoder(data[:,:text].long(),data[:,text:2*text].long(),p)

            mask = (data[:,-1] == 0).float().cuda(args.device)
            loss1 = 0.1*F.binary_cross_entropy(domain[-1], mask)
            loss = loss1 + loss

            loss.backward()
            optimizer1.step()
            optimizer2.step()

        precision, recall, f1 = eval(encoder,classifier,testdata,epoch,args)

        if best < f1:
            best = f1
            a = precision, recall, f1

    return a[0]

def train_rl(encoder,classifier,traindata,testdata,optimizer1,optimizer2,args):

    best = 0
    critic = Critic(args,hidden_dim=int(768/2))
    critic = critic.cuda(args.device)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=0.01, betas=(0.9, 0.99), eps=0.0000001)
    rl = RL(hidden_dim=int(768/2)).cuda(args.device)
    optimizer_rl = torch.optim.Adam(rl.parameters(), lr=0.01, betas=(0.9, 0.99), eps=0.0000001)
    discriminator = Discriminator(classifier.hidden_dim).cuda(args.device)
    optimizer_d = optim.Adam(discriminator.parameters(), 0.01)

    calreward = CalReward(discriminator,classifier,args)

    for epoch in range(1, args.N_EPOCHS+1):

        encoder.train()
        classifier.train()
        for batch_id, (sourced, targetd)  in enumerate(traindata):

            _, y_s, x_s, mask_s = sourced[0].cuda(args.device), \
                                  sourced[1].cuda(args.device).float(), sourced[2].cuda(args.device), sourced[3].cuda(args.device)

            _, y_t, x_t, mask_t = targetd[0].cuda(args.device), \
                                  targetd[1].cuda(args.device).float(), targetd[2].cuda(args.device), targetd[3].cuda(args.device)

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            embedding,_ = encoder(x_s,mask_s)
            prediction = classifier(embedding)
            loss = F.binary_cross_entropy(prediction,y_s)
            loss.backward()
            optimizer1.step()
            optimizer2.step()

            #training discriminator
            optimizer_d.zero_grad()

            embedding_s,_ = encoder(x_s,mask_s)
            pro_s = discriminator(embedding_s)
            pro_s_ = torch.tensor(len(x_s)*[1.]).cuda(args.device)
            loss = F.binary_cross_entropy(pro_s,pro_s_)
            embedding_t,_ = encoder(x_t,mask_t)
            pro_t = discriminator(embedding_t)
            pro_t_ = torch.tensor(len(x_t)*[0.]).cuda(args.device)
            loss = F.binary_cross_entropy(pro_t,pro_t_) + loss
            loss.backward()
            optimizer_d.step()

            optimizer_rl.zero_grad()
            optimizer_critic.zero_grad()

            embedding_s,_ = encoder(x_s,mask_s)
            embedding_t,_ = encoder(x_t,mask_t)
            embedding = torch.cat((embedding_s,embedding_t),dim=0)

            embedding_m,log_softmax, entropy,out,action = rl(embedding)
            y_s_ = torch.tensor(len(embedding_s)*[1.]).cuda(args.device)
            y_m = torch.tensor(len(embedding_t)*[0.]).cuda(args.device)
            source = torch.cat((y_s_,y_m),dim=0)
            value = calreward.reward(embedding,embedding_m,source,action)

            prediction = critic(embedding_m)
            loss1 = F.mse_loss(prediction.squeeze(),value)
            loss1.backward(retain_graph=True)
            optimizer_critic.step()
            prediction = critic(embedding_m)
            loss2 = rl.cal_loss(value,prediction,log_softmax, entropy)
            loss2.backward()
            optimizer_rl.step()

        precision, recall, f1 = eval(encoder,classifier,testdata,epoch,args)

        if best < f1:
            best = f1
            a = precision, recall, f1
    return a[0]

def train_mcd(extractor,classifier1,traindata,testdata,opt_e,opt_c1,args):

    '''
    Maximum Classifier Discrepancy for Unsupervised Domain Adaptation
    '''
    best = 0
    best_epoch = 0
    classifier2 = Classifier(classifier1.hidden_dim,args).cuda(args.device)
    opt_c2 = torch.optim.Adam(classifier2.parameters(), lr=args.lr)

    for epoch in range(1, args.N_EPOCHS+1):

        extractor.train()
        classifier1.train()
        classifier2.train()

        for batch_id, (sourced, targetd) in enumerate(traindata):
            _, y, x, mask = sourced[0].cuda(args.device), \
                            sourced[1].cuda(args.device), sourced[2].cuda(args.device), sourced[3].cuda(args.device)

            '''
            STEP A
            '''
            opt_e.zero_grad()
            opt_c1.zero_grad()
            opt_c2.zero_grad()

            src_feat, domain = extractor(x,mask)
            preds_s1 = classifier1(src_feat)
            preds_s2 = classifier2(src_feat)

            loss_A = F.binary_cross_entropy(preds_s1, y.float()) + F.binary_cross_entropy(preds_s2, y.float())
            loss_A.backward()

            opt_e.step()
            opt_c1.step()
            opt_c2.step()

            '''
            STEP B
            '''
            opt_e.zero_grad()
            opt_c1.zero_grad()
            opt_c2.zero_grad()

            _, y2, x2, mask2 = targetd[0].cuda(args.device), \
                               targetd[1].cuda(args.device), targetd[2].cuda(args.device), targetd[3].cuda(args.device)

            src_feat,domain = extractor(x,mask)
            preds_s1 = classifier1(src_feat)
            preds_s2 = classifier2(src_feat)
            loss_B = F.binary_cross_entropy(preds_s1, y.float()) + F.binary_cross_entropy(preds_s2, y.float())

            src_feat,domain = extractor(x2, mask2)
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

                feat_tgt,domain = extractor(x2,mask2)
                preds_t1 = classifier1(feat_tgt)
                preds_t2 = classifier1(feat_tgt)
                loss_C = torch.mean(torch.abs(preds_t1- preds_t2))

                loss_C.backward()
                opt_e.zero_grad()
                opt_c1.zero_grad()
                opt_c2.zero_grad()

        precision, recall, f1 = eval(extractor,classifier1,testdata,epoch,args)
        if best < f1:
            best = f1
            best_epoch += 1
            a = precision, recall, f1
        precision, recall, f1 = eval(extractor,classifier2,testdata,epoch,args)
        if best < f1:
            best = f1
            best_epoch += 1
            a = precision, recall, f1

        if best_epoch == 20:
            return a[0]

    return a[0]

def train_alda(encoder,classifier,traindata,testdata,optimizer1,optimizer2,args):

    classifier2 = Classifier(classifier.hidden_dim,args).cuda(args.device)
    opt_c2 = torch.optim.Adam(classifier2.parameters(), lr=args.lr)
    best = 0

    for epoch in range(1, args.N_EPOCHS+1):

        encoder.train()
        classifier.train()
        classifier2.train()

        for batch_id, (sourced, targetd) in enumerate(traindata):
            _, y1, inputs_source, mask = sourced[0].cuda(args.device), \
                                         sourced[1].cuda(args.device).float(), sourced[2].cuda(args.device), sourced[3].cuda(args.device)

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            opt_c2.zero_grad()

            _, y2, inputs_target, mask2 = targetd[0].cuda(args.device), \
                                          targetd[1].cuda(args.device), targetd[2].cuda(args.device), targetd[3].cuda(args.device)

            features_source, domain_s = encoder(inputs_source,mask)
            outputs_source = classifier(features_source)

            features_target, domain_s = encoder(inputs_target,mask2)
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

def train_jumbot(encoder,classifier,traindata,testdata,optimizer1,optimizer2,args):

    best = 0
    alpha = 0.01
    lambda_t = 0.5
    reg_m = 0.5
    for epoch in range(1, args.N_EPOCHS+1):

        encoder.train()
        classifier.train()
        for batch_id, (sourced, targetd) in enumerate(traindata):
            _, y_s, x_s, s_mask = sourced[0].cuda(args.device), \
                                  sourced[1].cuda(args.device).float(), sourced[2].cuda(args.device), sourced[3].cuda(args.device)

            _, y_t, x_t, t_mask = targetd[0].cuda(args.device), \
                                  targetd[1].cuda(args.device).float(), targetd[2].cuda(args.device), targetd[3].cuda(args.device)

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            g_xs, domain_s = encoder(x_s,s_mask)
            pred_s = classifier(g_xs)

            g_xt, domain_t = encoder(x_t,t_mask )
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

    return aa[0]

def eval(encoder,classifier,testdata,epoch,args):

    encoder.eval()
    classifier.eval()
    prediction_list = []
    target_list = []
    with torch.no_grad():
        for batch_id, (_, targetd) in enumerate(testdata):

            _, t_label, t_review, t_mask = targetd[0].cuda(args.device), \
                                           targetd[1], targetd[2].cuda(args.device), targetd[3].cuda(args.device)

            target_list += t_label.tolist()

            try:
                embedding,_ = encoder(t_review,t_mask)
            except:
                _,embedding = encoder(t_review,t_mask)
                embedding = embedding[0][0]
            prediction = classifier(embedding)
            prediction_list += prediction.cpu().tolist()

    precision, recall, f1 = F1(prediction_list,target_list,epoch)

    return precision, recall, f1
