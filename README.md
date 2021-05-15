The sentiment data consist of 1,000 Amazon reviews that have been coded to have a positive (1) or negative sentiment (0). The data are called “Amazon labelled text” and are available from UCI’s machine learning repository http://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences.

The data is split into training/test data. Predicting using a Random Forest algorithm with default values of creating n-gram variables as follows: with stemming, removing stopwords, setting threshold to 5, for the tuning parameters. 

Comparing prediction accuracies for count n-gram variables with indicator n-gram variables for the thresholds: 3, 5 ,7, 9. 

Also comparing prediction accuracies for the four combinations (with/ without) stemming and stopwords. 

Running a logistic regression on the default n-gram encoding.
