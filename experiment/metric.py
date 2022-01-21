import numpy as np
from loguru import logger

def F1(prediction,target,epoch):

    prediction = np.array((np.array(prediction)>0.5),dtype=float)
    target = np.array(target)

    TP = (prediction*target).sum()
    precision = TP/prediction.sum()
    recall = TP/target.sum()
    f1 = 2 * (precision * recall)/(precision+recall)

    f1 = round(f1, 6)
    logger.info('test epoch: {}, recall: {}, precision:{}, F1:{}'.format(epoch,recall,precision,f1))
    return precision, recall, f1