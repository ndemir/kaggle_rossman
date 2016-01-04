# kaggle_rossman

##Script for Rossman Store Sales Competititon on Kaggle

I used xgboost as the most of the competititors in this [competitition](https://www.kaggle.com/c/rossmann-store-sales/leaderboard) did.
Before running xgboost, I clustered the data to bring the similar stores together, that helped me to build better regressors.
I built 100 clusters and trained a regressor for each cluster.

I put submission.csv in the output directory.
[Submitting](https://www.kaggle.com/c/rossmann-store-sales/submissions/attach) this file produces the score 0.11695 
which is in the top 10% of the leaderboard.
