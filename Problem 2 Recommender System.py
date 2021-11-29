"Problem 2"

import pandas as pd

# import Dataset 
Entertainment = pd.read_csv("D:/C DRIVE-SSD DATA backup 15-12-2020/Desktop/360DigiTmg Assignment/Recommender System/Entertainment.csv", encoding = 'utf8')
Entertainment.shape # shape
Entertainment.columns
Entertainment.Category # genre columns

#Exploratory Data Analysis
# Check the descriptive statistics of numeric variables
Entertainment.describe()
#1st moment Business Decision # Measures of Central Tendency / First moment business decision
Entertainment.mean()
Entertainment.median()
Entertainment.mode()

# Measure of Dispersion / Second moment business decision
Entertainment.var() #variance
Entertainment.std() #standard variance

# Third moment business decision
Entertainment.skew()

# Fourth moment business decision
Entertainment.kurt()

# Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes
import numpy as np

plt.hist(Entertainment.Reviews) #histogram
help(plt.hist)

plt.boxplot(Entertainment.Reviews, vert=False)

from sklearn.feature_extraction.text import TfidfVectorizer #term frequencey- inverse document frequncy is a numerical statistic that is intended to reflect how important a word is to document in a collecion or corpus
# Tfidf is calculate numeric weightages for the data that we are going to generate

# Creating a Tfidf Vectorizer to remove all stop words
tfidf = TfidfVectorizer(stop_words = "english")    # taking stop words from tfidf vectorizer

# replacing the NaN values in overview column with empty string
Entertainment["Category"].isnull().sum() 
Entertainment["Category"] = Entertainment["Category"].fillna(" ") # fill with space (empty string)
#we can fill mode in place of missing values

# Preparing the Tfidf matrix by fitting and transforming
tfidf_matrix = tfidf.fit_transform(Entertainment.Category)   #Transform a count matrix to a normalized tf or tf-idf representation
# It converts columns into one hot encoding
tfidf_matrix.shape #12294, 34 (34 unique genres is there, that means type of movies)

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

# most similar name of movie
# creating a mapping of Entertainment Titles to index number
#each movie has a index number, we are trying to capture index based on the Tiles
Entertainment_index = pd.Series(Entertainment.index, index = Entertainment['Titles']).drop_duplicates()

# it gives index number
Entertainment_id = Entertainment_index["Toy Story (1995)"]
Entertainment_id

def get_recommendations(Name, topN):    
    # topN = 10
    # Getting the movie index using its title 
    Entertainment_id = Entertainment_index[Name]
    
    # Getting the pair wise similarity score for all the Entertainment's with that 
    # Entertainment
    cosine_scores = list(enumerate(cosine_sim_matrix[Entertainment_id]))
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores = sorted(cosine_scores, key=lambda x:x[1], reverse = True) #decreasing order
    
    # Get the scores of top N most similar movies 
    cosine_scores_N = cosine_scores[0: topN+1]
    
    # Getting the movie index 
    Entertainment_idx  =  [i[0] for i in cosine_scores_N]
    Entertainment_scores =  [i[1] for i in cosine_scores_N]
    
    # Similar movies and scores
    Entertainment_similar_show = pd.DataFrame(columns=["Titles", "Score"])
    Entertainment_similar_show["Titles"] = Entertainment.loc[Entertainment_idx, "Titles"]
    Entertainment_similar_show["Score"] = Entertainment_scores
    Entertainment_similar_show.reset_index(inplace = True)  
    # Entertainment_similar_show.drop(["index"], axis=1, inplace=True)
    print (Entertainment_similar_show)
    # return (Entertainment_similar_show)

    
# Enter your Entertainment and number of Entertainment's to be recommended 
get_recommendations("Casino (1995)", topN = 10) #similar type of movies, same jondras
Entertainment_index["Casino (1995)"]







