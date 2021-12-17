# EECS-595-Support-and-Sentiment-COVID19Vaccine-Tweets
This repository provides the code to reproduce the results of the EECS 595 NLP project "Support and Sentiment in COVID-19 Vaccine Tweets Across America"

To access the analyzed data please reach out to the project members.

## Data preprocessing
To filter the COVID-19 tweets for geotaggs and addressing vaccines execute the data_cleaning.py script from the data directory.
Unhydrated twitter data ids can be found in the data folder, we cannot release the hydrated tweets due to twitters privacy policy
To filter for CT-BERT go to https://github.com/digitalepidemiologylab/covid-twitter-bert
To filter for VADER go to https://github.com/cjhutto/vaderSentiment.git

## Running Models
The CT-BERT fine tuning, execution, and analysis is all in CT-BERT/TwitterPull.ipynb

# Sentiment analysis with VADER
The sentiment analysis is performed by the VADER implemention of Hutto, C.J. and Gilbert, E.E (Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.). 
To reproduce the results presented in the project report execude the vader_analysis.py.
