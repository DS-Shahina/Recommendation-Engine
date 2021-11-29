"Problem 1"

# import os
import pandas as pd

# import Dataset 
game = pd.read_csv("D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360DigiTmg Assignment/Recommender System/game.csv", encoding = 'utf8')
game.shape # shape
game.columns
game.game # genre columns

#Exploratory Data Analysis
# Check the descriptive statistics of numeric variables
game.describe()
#1st moment Business Decision # Measures of Central Tendency / First moment business decision
game.mean()
game.median()
game.mode()

# Measure of Dispersion / Second moment business decision
game.var() #variance
game.std() #standard variance

# Third moment business decision
game.skew()

# Fourth moment business decision
game.kurt()

# Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes
import numpy as np

plt.hist(game.rating) #histogram
help(plt.hist)

plt.boxplot(game.rating, vert=False)

from sklearn.feature_extraction.text import TfidfVectorizer #term frequencey- inverse document frequncy is a numerical statistic that is intended to reflect how important a word is to document in a collecion or corpus
# Tfidf is calculate numeric weightages for the data that we are going to generate

# Creating a Tfidf Vectorizer to remove all stop words
tfidf = TfidfVectorizer(stop_words = "english")    # taking stop words from tfidf vectorizer

# Checking is there any null values is there
game["game"].isnull().sum() 

# Preparing the Tfidf matrix by fitting and transforming
tfidf_matrix = tfidf.fit_transform(game.game)   #Transform a count matrix to a normalized tf or tf-idf representation
# It converts columns into one hot encoding
tfidf_matrix.shape #12294, 3068 (3068 unique game is there)

# with the above matrix we need to find the similarity score
# There are several metrics for this such as the euclidean, 
# the Pearson and the cosine similarity scores

# For now we will be using cosine similarity matrix
# A numeric quantity to represent the similarity between 2 movies 
# Cosine similarity - metric is independent of magnitude and easy to calculate 

# cosine(x,y)= (x.y⊺)/(||x||.||y||)

from sklearn.metrics.pairwise import linear_kernel
# Computing the cosine similarity on Tfidf matrix
cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix) #it measuring row to row matrix & it is same matrix, diagonal is 1
#1st row of 1st matrix to multiply with all row of 2nd matrix because both are same matrix
#this similarity matrix calculated offline made ready

# most similar name of game
# creating a mapping of game name to index number
#each game has a index number, we are trying to capture index based on the name
game_index = game.drop_duplicates(subset=["game"])
game_index = pd.Series(game_index.index, index = game_index['game'])
# it gives index number
game_id = game_index["Super Mario Galaxy"]
game_id

def get_recommendations(Name, topN):    
    # topN = 10
    # Getting the game index using its title 
    game_id = game_index[Name]
    
    # Getting the pair wise similarity score for all the game's with that 
    # game
    cosine_scores = list(enumerate(cosine_sim_matrix[game_id]))
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores = sorted(cosine_scores, key=lambda x:x[1], reverse = True) #decreasing order
    
    # Get the scores of top N most similar movies 
    cosine_scores_N = cosine_scores[0: topN+1]
    
    # Getting the game index 
    game_idx  =  [i[0] for i in cosine_scores_N]
    game_scores =  [i[1] for i in cosine_scores_N]
    
    # Similar games and scores
    game_similar_show = pd.DataFrame(columns=["game", "Score"])
    game_similar_show["game"] = game.loc[game_idx, "game"]
    game_similar_show["Score"] = game_scores
    game_similar_show.reset_index(inplace = True)  
    # game_similar_show.drop(["index"], axis=1, inplace=True)
    print (game_similar_show)
    # return (game_similar_show)

    
# Enter your game and number of game's to be recommended
get_recommendations("Mass Effect", topN = 10) #similar type of games, 
game_index["Mass Effect"]











