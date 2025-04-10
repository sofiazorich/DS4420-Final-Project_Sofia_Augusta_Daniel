import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import random
 

# function to read in data
def process_data(data_path):
    data = pd.read_csv(data_path)
    feature_cols = ['elevation_difference_m', 'total_slope_length_km', 'number_of_lifts', 'annual_snowfall_cm']
    data = data[['name'] + feature_cols].set_index('name')
    scaler = StandardScaler()
    data[feature_cols] = scaler.fit_transform(data[feature_cols])
    return data

def simulate_user_ratings(feature_df, num_resorts_to_rate):
    # get list of resorts from dataframe index
    resorts = list(feature_df.index)
    # empty dictionary to store user ratings
    user_ratings = {}
    # simulate random ratings for each resort 
    for resort in resorts:
        user_ratings[resort] = random.randint(1, 5)
    return user_ratings

def predict_user_rating(feature_df, user_ratings):
    # get list of resorts from dataframe index
    resorts = list(feature_df.index)
    # get unrated resorts
    unrated_resorts = [resort for resort in resorts if resort not in user_ratings]
    # pick a random unrated resort to predict rating of
    target = random.choice(unrated_resorts)
    # compute the similarity between other resorts
    similarity_df = pd.DataFrame(
        cosine_similarity(feature_df),
        index=feature_df.index,
        columns=feature_df.index
    )
    # get the similarities between the target resort and the resorts that the user has rated
    # turn user_ratings into series for easier manipulation
    user_ratings = pd.Series(user_ratings)

    # get the similarities between the target resort and the resorts that the user has rated
    user_rated_resorts_sims = similarity_df.loc[target, user_ratings.index]
    # get user ratings for those resorts
    ratings = user_ratings[user_rated_resorts_sims.index]
    # get weighted average with ratings and cosine similarity scores
    term1 = np.sum(user_rated_resorts_sims * ratings)
    term2 = np.sum(np.abs(user_rated_resorts_sims))
    # produce rating
    prediction = term1 / term2
    print(f"\nPredicted user rating for '{target}': {prediction:.4f}")


