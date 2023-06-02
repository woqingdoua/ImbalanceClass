# ImbalanceClass
# UDA

This is the implementation of [Unsupervised Reinforcement Adaptation for Class-Imbalanced Text Classification](https://aclanthology.org/2022.starsem-1.27/) at 
*SEM-2022.

## Citations

If you use or extend our work, please cite our paper at *SEM-2022.
```
@inproceedings{wu-huang-2022-unsupervised,
    title = "Unsupervised Reinforcement Adaptation for Class-Imbalanced Text Classification",
    author = "Wu, Yuexin  and
      Huang, Xiaolei",
    booktitle = "Proceedings of the 11th Joint Conference on Lexical and Computational Semantics",
    month = jul,
    year = "2022",
    address = "Seattle, Washington",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.starsem-1.27",
    doi = "10.18653/v1/2022.starsem-1.27",
    pages = "311--322",
    abstract = "Class imbalance naturally exists when label distributions are not aligned across source and target domains. However, existing state-of-the-art UDA models learn domain-invariant representations across domains and evaluate primarily on class-balanced data. In this work, we propose an unsupervised domain adaptation approach via reinforcement learning that jointly leverages feature variants and imbalanced labels across domains. We experiment with the text classification task for its easily accessible datasets and compare the proposed method with five baselines. Experiments on three datasets prove that our proposed method can effectively learn robust domain-invariant representations and successfully adapt text classifiers on imbalanced classes over domains.",
}
```

## Datasets
We use three datasets (Amazon, Yelp and IMDB Review) in our paper.

## Train

Run `python main.py`.


## Contacts
Because the experimental datasets are too large to share all of them. Please send any requests or questions to my email: [ywu10@memphis.edu](ywu10@memphis.edu).

 
