# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 07:39:13 2025

@author: michael.olson2
"""

import numpy as np
import pandas as pd

###  We'll need the cosine distance
def cosine_distance(vector_1,vector_2):
    dotproduct = np.dot(vector_1, vector_2)
    norms = np.dot(vector_1, vector_1)*np.dot(vector_2, vector_2)
    return dotproduct/np.sqrt(norms)


#####  Utility Matrix  #####

## Binary (Like/Dislike) ratings

movies = ['M01','M02','M03','M04','M05','M06','M07','M08','M09','M10','M11','M12','M13','M14','M15']

likes = {
    'User A' : [0,1,1,1,0,1,1,0,1,0,0,1,1,1,1],
    'User B' : [1,1,0,0,0,0,1,1,0,1,0,1,0,0,1],
    'User C' : [0,0,0,1,1,0,0,0,0,1,0,0,0,0,1]
}

user_likes = pd.DataFrame(likes).transpose()
user_likes.columns = movies

print(user_likes)


## Scaled (5-star) ratings

ratings = {
    'User A' : [0,1,2,1,0,4,4,0,3,0,0,5,5,4,5],
    'User B' : [3,4,0,0,0,0,1,5,0,3,0,5,0,0,4],
    'User C' : [0,0,0,2,4,0,0,0,0,4,0,0,0,0,3]
}

user_ratings = pd.DataFrame(ratings).transpose()
user_ratings.columns = movies

print(user_ratings)



#####  Item Profile  #####

## Item Profile : Movie Database
actors_list = ['Julia Roberts','Robin Williams','Clint Eastwood','Ian McKellen','Movie Rating']

actors = {
    'M01' : [1,0,0,1,3],
    'M02' : [0,0,1,0,5],
    'M03' : [1,0,0,0,4],
    'M04' : [0,1,0,0,2],
    'M05' : [0,1,0,0,4],
    'M06' : [0,0,1,0,4],
    'M07' : [1,1,0,0,3],
    'M08' : [1,0,0,1,1],
    'M09' : [0,0,1,1,5],
    'M10' : [1,1,0,0,5],
    'M11' : [1,0,0,0,1],
    'M12' : [0,0,0,1,2],
    'M13' : [0,1,0,1,2],
    'M14' : [0,1,1,0,5],
    'M15' : [0,0,1,1,2]
}

movie_casts = pd.DataFrame(actors).transpose()
movie_casts.columns = actors_list

print(movie_casts)


## Comparing two movies with a scaling factor - Similar to Example 9.2
alpha = 2
movie_1 = 'M07'
movie_2 = 'M10'

movie_comparison = movie_casts.loc[[movie_1, movie_2]]
movie_comparison['Movie Rating'] = movie_comparison['Movie Rating']*alpha
print(movie_comparison)

## Find the distance!
cosine_distance(movie_comparison.loc[movie_1], movie_comparison.loc[movie_2])

## Calculating cosine distance between all movies
alpha = 0.5
movie_comparison = movie_casts.copy()
movie_comparison['Movie Rating'] *= alpha

distances = pd.DataFrame(columns=movies)
for movie1 in movies:
    for movie2 in movies:
        distances.loc[movie1,movie2] = cosine_distance(movie_comparison.loc[movie1],movie_comparison.loc[movie2])

print(distances)



#####  User Profile  #####

## User Profile - binary ratings
## Compare with Example 9.3 : Movies given only likes or dislikes
dotproduct = np.dot(movie_casts['Julia Roberts'], user_likes.loc['User A'])
norm = np.dot(user_likes.loc['User A'], user_likes.loc['User A'])
print(f"Dot Product of User Likes and Julia Roberts' Movies : {dotproduct}")
print(f"Norm of User Likes (number of movies liked)         : {norm}")
print(f"User weight to movies with Julia Roberts            : {dotproduct / norm}\n")


user_profile_likes = pd.DataFrame(columns=actors_list)

for actor_id in actors_list:
    for user_id in user_ratings.index:
        user_profile_likes.loc[user_id,actor_id] = np.dot(movie_casts[actor_id], user_ratings.loc[user_id]) / user_ratings.loc[user_id].sum()
        
print(user_profile_likes)
#user_profile_likes.drop('Movie Rating', axis=1)


print(user_ratings)


## Average Rating given by user over all videos - Compare with Example 9.4 : Movies given scaled ratings

# Replace all 0's with NaN so they don't influence the average
print(user_ratings.replace(0,np.nan).loc['User A'].mean())

user_avg_rating = user_ratings.replace(0,np.nan).loc['User A'].mean()
print(f"Average rating by this user = {user_avg_rating}")


### Next find the movies with a given actor and the ratings the user has given it
actor_ratings_from_user = movie_casts['Julia Roberts'] * user_ratings.loc['User A']
print(actor_ratings_from_user)


### Normalize the ratings by subtracting the average rating
actor_ratings_from_user = actor_ratings_from_user.apply(lambda x: 0 if x==0 else x-user_avg_rating)
print(actor_ratings_from_user)

### The average is the score
print(actor_ratings_from_user.replace(0,np.nan).mean())

### Do for all
a = 'Julia Roberts'
u = 'User A'

avg_rating = user_ratings.replace(0,np.nan).loc[u].mean()
print((movie_casts[a] * user_ratings.loc[u]).apply(lambda x: 0 if x==0 else x-avg_rating).replace(0,np.nan).mean())

## User Profile - Apply for all users and actors
user_profile_ratings = pd.DataFrame(columns=actors_list)

for actor_id in actors_list:
    for user_id in user_ratings.index:
        avg_rating = user_ratings.replace(0,np.nan).loc[user_id].mean()  # Average rating given by user
        tmp = movie_casts[actor_id] * user_ratings.loc[user_id]          # Array of ratings given by user involving the given actor 
        user_profile_ratings.loc[user_id, actor_id] = tmp.apply(lambda x: 0 if x==0 else x-avg_rating).replace(0,np.nan).mean() # Subtract avg rating from ratings given, then take the mean
        

user_profile_ratings.drop('Movie Rating', axis=1, inplace=True)
print(user_profile_ratings)



