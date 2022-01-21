import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Analysis:
    def __init__(self,file='/home/ywu10/Documents/MoralCausality/data/MFTC_V4_text_preprogress.tsv',ratio=True):
        self.data = pd.read_csv(file)
        self.ratio = ratio
        self.distribution()

    def distribution(self):
        self.source = list(set(self.data['source'].tolist()))
        label = self.data['label'].tolist()
        self.label = list(set(sum([l.split(',') for l in label],[])))
        self.dist = dict.fromkeys(self.source)
        for ss in self.source:
            ll = self.data[self.data['source'] == ss]['label'].tolist()
            ll = list(sum([l.split(',') for l in ll],[]))
            b = []
            for l in self.label:
                b.append(ll.count(l))

            if self.ratio == True:
                self.dist[ss] = np.array(b)/sum(b)

        dist = []
        for i in self.dist:
            dist.append(self.dist[i])

        return self.dist, self.label, self.source

    def paintdistribution(self):

        self.distribution()
        fig, axs = plt.subplots(1, len(self.dist), figsize=(60, 8), sharey=True)
        index = 0
        for ss in self.dist:
            axs[index].bar(self.label, self.dist[ss])
            index += 1
        fig.suptitle('Categorical Plotting')
        plt.savefig('label distribution.jpg')

