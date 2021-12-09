# Session-based Recommendation Library

## MSGIFSR

This is the implementation of **Learning Multi-granularity User Intent Unit for Session-based Recommendation** from WSDM 2022 and some other session-based recommendation baselines. We mainly follow the implementation of lessr. (Tianwen Chen, Raymond Wong, KDD 2020)


## Baselines

We also reimplemented several current session-based recommendation baselines and tuned them through our best effert.

## Dataset

Download and extract the following datasets and put the files in the dataset folder named under datasets/$DATASETNAME

* [Diginetica](https://competitions.codalab.org/competitions/11161#learn_the_details-data2)
* [Gowalla](http://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz)
* [LastFM](http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html)
* [Yoochoose](https://www.kaggle.com/chadgostopp/recsys-challenge-2015)

Then run the code in src/utils/data/preprocess to process them.

## Run

<p>python main_ccsgnn.py --dataset-dir datasets/$DATASETNAME</p>


