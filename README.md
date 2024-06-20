# Session-based Recommendation Library

## MSGIFSR

This is the official implementation of **Learning Multi-granularity User Intent Unit for Session-based Recommendation** from WSDM 2022 and some other session-based recommendation models. We use [DGL](https://www.dgl.ai/) library and mainly follow the implementation of [lessr](https://github.com/twchen/lessr). (Tianwen Chen, Raymond Wong, KDD 2020)

## Baselines

We also reimplemented several current session-based recommendation baselines including SRGNN, NISER+ and tuned them through our best effert. They are summarized as follows. Leaderboards are comming soon.

## Dataset

Download and extract the following datasets and put the files in the dataset folder named under datasets/$DATASETNAME

* [Diginetica](https://competitions.codalab.org/competitions/11161#learn_the_details-data2)
* [Gowalla](http://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz)
* [LastFM](http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html)
* [Yoochoose](https://www.kaggle.com/chadgostopp/recsys-challenge-2015)

Then run the code in src/utils/data/preprocess to process them. When processing yoochoose, please run preprocess_yoochoose.py in datasets first and then run the code in src/utils/data/preprocess.

## Run

first create the enveriment.

```
conda env create -f environment.yaml
```

then 

```
bash start.sh $MODEL_NAME $DATASET_NAME
```
## Experiment Results

We find that keeping the original order of training data makes the result better. It is due to the way of splitting the dataset. Current public session-based recommendation datasets usually split train/test data according to time. This will make the distribution of samples at the latter positions of the training data more similar to the test data than those at the former positions. Without shuffling the model will fit better. This is a common phemonemon in recommender systems since user interest evolves fast and too early samples will not help the recommendation. We also provide a version that shuffle the training dataset. In both settings the testing dataset is shuffled.

## Citation

```
@inproceedings{10.1145/3488560.3498524,
      author = {Jiayan Guo and Yaming Yang and Xiangchen Song and Yuan Zhang and Yujing Wang and Jing Bai and Yan Zhang},
      title = {Learning Multi-granularity User Intent Unit for Session-based Recommendation},
      year = {2022},
      booktitle = {Proceedings of the 15'th ACM International Conference on Web Search and Data Mining {WSDM}'22},
}
```


