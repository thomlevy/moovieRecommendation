# moovieRecommendation
This project does moovie recommendation with collaborative filtering using information about :
* users
* movies 
* preferences of some users towards some movies

Using this recommendation system, it allows predicting preferences of users towards movies for new (user,movie) pairs.

## Context
The recommendation relies on moovie ratings that includes more than 43k moovies and 270k users.
I tried to predict preferences with a model based approach using collaborative filtering library from [Surprise library](https://surprise.readthedocs.io/en/stable/index.html)
However, this number of users doesn't fit into Memory.
One option could have been to do distributed computing with a cluster of nodes (e.g with Spark) but I don't have a cluster available.
As a simplification to scale the project, I decided to split the ratings into chunks of consecutive users.
Thus, the following steps are done:
* Split of the ratings/eval_ratings into chunks: Here, I choosed a chunkSize=13545 to have 20 chunks (so that the last chunk has approx the same nb of users than the other ones).
* For each chunk, learn the model from known ratings and predict preferences for new (user,movie) pairs.
* Merge the predictions into a single `evaluation_ratings_out.csv` flie.

For the predition, I used Singular Value Decomposition (SVD) algo that appears to provide realtively good accuracy when doing cross validation.

## How to install
* Clone this repository
* Requires Python 3.6 (or higher): tested only with 3.6
* Requires [Surprise library](https://surprise.readthedocs.io/en/stable/index.html) : To install `pip install scikit-surprise`

## How to run
* `python moovieReco.py`
NOTE: You can change the number of epochs (default=20) to 1 if you want to speed-up the learning (but with lower precision)

## How to test
* Basic unit testing of the functions can be done with: `python UnitTest.py -v` 