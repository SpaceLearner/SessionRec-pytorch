# Session-based Recommendation Library

## MSGIFSR

This is the implementation of **Learning Multi-granularity User Intent Unit for Session-based Recommendation** from WSDM 2022 and some other session-based recommendation models. We mainly follow the implementation of [lessr](https://github.com/twchen/lessr). (Tianwen Chen, Raymond Wong, KDD 2020)


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

<p>bash start.sh $MODEL_NAME $DATASET_NAME</p>

## Experiment Results

We find that keeping the original order of training data makes the result better. It is due to the way of splitting the data. Current public session-based recommendation datasets usually split data according to time. This will make the distribution of the last few samples of the training data similar to the test data than samples that at the former positions. Without shuffling the model fits better.


