# UDA

This is the implementation of [A Gumbel-based Rating Prediction Framework for Imbalanced Recommendation](https://arxiv.org/pdf/2012.05009.pdf) at 
CIKM-2022.

## Citations

If you use or extend our work, please cite our paper at CHIL-2023.
```
@inproceedings{10.1145/3511808.3557341,
author = {Wu, Yuexin and Huang, Xiaolei},
title = {A Gumbel-Based Rating Prediction Framework for Imbalanced Recommendation},
year = {2022},
isbn = {9781450392365},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3511808.3557341},
doi = {10.1145/3511808.3557341},
abstract = {Rating prediction is a core problem in recommender systems to quantify users' preferences towards items. However, rating imbalance naturally roots in real-world user ratings that cause biased predictions and lead to poor performance on tail ratings. While existing approaches in the rating prediction task deploy weighted cross-entropy to re-weight training samples, such approaches commonly assume a normal distribution, a symmetrical and balanced space. In contrast to the normal assumption, we propose a novel Gumbel-based Variational Network framework (GVN) to model rating imbalance and augment feature representations by the Gumbel distributions. We propose a Gumbel-based variational encoder to transform features into non-normal vector space. Second, we deploy a multi-scale convolutional fusion network to integrate comprehensive views of users and items from the rating matrix and user reviews. Third, we adopt a skip connection module to personalize final rating predictions. We conduct extensive experiments on five datasets with both errors- and ranking-based metrics. Experiments on ranking and regression evaluation tasks prove that the GVN can effectively achieve state-of-the-art performance across the datasets and reduce the biased predictions of tail ratings. We compare with various distributions (e.g., normal and Poisson) and demonstrate the effectiveness of Gumbel-based methods on class-imbalance modeling. The code is available at https://github.com/woqingdoua/Gumbel-recommendation-for-imbalanced-data.},
booktitle = {Proceedings of the 31st ACM International Conference on Information &amp; Knowledge Management},
pages = {2199–2209},
numpages = {11},
keywords = {recommender system, neural networks, imbalanced distribution},
location = {Atlanta, GA, USA},
series = {CIKM '22}
}

```

## Datasets
We use three datasets (Amazon, Yelp and IMDB Review) in our paper.

## Train

Run `python main.py`.


## Contacts
Because the experimental datasets are too large to share all of them. Please send any requests or questions to my email: [ywu10@memphis.edu](ywu10@memphis.edu).
