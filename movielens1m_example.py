import openai
from openai import OpenAI
import google.generativeai as genai
import pandas as pd
import json
from sklearn.metrics import root_mean_squared_error, precision_score, recall_score, accuracy_score
from tqdm import tqdm
import time
import os
import re
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, cohen_kappa_score
from scipy.stats import spearmanr
from dotenv import load_dotenv
load_dotenv()

# import Lusifer
from Lusifer import Lusifer


# path to the folder containing movielens data
Path = "/content/Lusifer/Samples/Data/100k"


# Use your actual API key securely
KEY = os.getenv('GOOGLE_API_KEY')

# --------------------------------------------------------------
def load_data():
    # Paths for the processed files


    processed_users_file = f"{Path}/users_with_summary_df.csv"
    processed_ratings_file = f"{Path}/rating_test_df_100k.csv"

    # loading users dataframe
    if os.path.exists(processed_users_file):
        # Load the processed files if they exist
        users_df = pd.read_csv(processed_users_file)
        # users_df = pd.read_pickle(processed_users_file)

    else:
        users_df = pd.read_pickle(f"{Path}/user_dataset.pkl")
        users_df = users_df[["user_id", "user_info"]]

    # loading ratings dataframe
    if os.path.exists(processed_ratings_file):
        rating_test_df = pd.read_csv(processed_ratings_file)

    else:
        rating_test_df = pd.read_csv(f'{Path}/u1.test', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'],
                                     encoding='latin-1')

    # loading ratings: Train set
    rating_df = pd.read_csv(f'{Path}/u1.base', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'],
                            encoding='latin-1')

    # Load movies
    movies_df = pd.read_pickle(f"{Path}/movies_enriched_dataset.pkl")
    movies_df = movies_df[["movie_id", "movie_info"]]

    # Add new column to store simulated ratings if it doesn't exist
    if 'simulated_ratings' not in rating_test_df.columns:
        rating_test_df['simulated_ratings'] = None

    if 'user_profile' not in users_df.columns:
        users_df['user_profile'] = None

    return movies_df, users_df, rating_df, rating_test_df


# --------------------------------------------------------------
def compare_ratings(user_id, llm_ratings, rating_test_df):
    predicted_ratings = pd.DataFrame.from_dict(llm_ratings, orient='index', columns=['predicted']).astype(float)
    actual_ratings = rating_test_df[rating_test_df['user_id'] == user_id][['movie_id', 'rating']].set_index(
        'movie_id').astype(float)
    comparison = actual_ratings.join(predicted_ratings, how='inner').dropna()
    comparison.columns = ['actual', 'predicted']
    comparison['error'] = comparison['actual'] - comparison['predicted']

    rmse = root_mean_squared_error(comparison['actual'], comparison['predicted'])
    precision = precision_score(comparison['actual'], comparison['predicted'], average='micro')
    recall = recall_score(comparison['actual'], comparison['predicted'], average='micro')
    accuracy = accuracy_score(comparison['actual'], comparison['predicted'].round())

    return rmse, precision, recall, accuracy


# --------------------------------------------------------------
def evaluate_result(dataframe):

    # Drop rows with NaNs in 'simulated_ratings' or 'rating'
    dataframe = dataframe.dropna(subset=['simulated_ratings', 'rating'])

    # Ensure ratings are integers
    dataframe['rating'] = dataframe['rating'].astype(int)
    dataframe['simulated_ratings'] = dataframe['simulated_ratings'].astype(int)

    # RMSE
    mse = mean_squared_error(dataframe['rating'], dataframe['simulated_ratings'])
    rmse = np.sqrt(mse)

    # MAE
    mae = mean_absolute_error(dataframe['rating'], dataframe['simulated_ratings'])

    # R² Score
    r2 = r2_score(dataframe['rating'], dataframe['simulated_ratings'])

    # Spearman's Rank Correlation Coefficient
    spearman_corr, spearman_pvalue = spearmanr(dataframe['rating'], dataframe['simulated_ratings'])

    # Cohen's Kappa (Quadratic Weighted)
    kappa = cohen_kappa_score(dataframe['rating'], dataframe['simulated_ratings'], weights='quadratic')

    # Difference between actual and simulated ratings
    difference = (dataframe['rating'] - dataframe['simulated_ratings']).abs()

    # Exact match
    exact_matches = difference == 0
    exact_match_count = exact_matches.sum()
    exact_match_percentage = exact_match_count / len(dataframe) * 100

    # Off by 1 level
    off_by_1 = difference == 1
    off_by_1_count = off_by_1.sum()
    off_by_1_percentage = off_by_1_count / len(dataframe) * 100

    # Off by more than 1 level
    off_by_more_than_1 = difference > 1
    off_by_more_than_1_count = off_by_more_than_1.sum()
    off_by_more_than_1_percentage = off_by_more_than_1_count / len(dataframe) * 100

    # Close match (Exact match + Off by 1 level)
    close_match_count = exact_match_count + off_by_1_count
    close_match_percentage = close_match_count / len(dataframe) * 100

    # Weighted accuracy
    weighted_accuracy = (exact_matches * 1 + off_by_1 * 0.8).sum() / len(dataframe)

    # Output the results
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Spearman's Rank Correlation Coefficient: {spearman_corr:.4f} (p-value: {spearman_pvalue:.4f})")
    print(f"Cohen's Kappa (Quadratic Weighted): {kappa:.4f}\n")

    print(f"Exact Match Count: {exact_match_count}")
    print(f"Exact Match Percentage: {exact_match_percentage:.2f}%\n")

    print(f"Off by 1 Level Count: {off_by_1_count}")
    print(f"Off by 1 Level Percentage: {off_by_1_percentage:.2f}%\n")

    print(f"Off by More Than 1 Level Count: {off_by_more_than_1_count}")
    print(f"Off by More Than 1 Level Percentage: {off_by_more_than_1_percentage:.2f}%\n")

    print(f"Close Match Count (Exact + Off by 1): {close_match_count}")
    print(f"Close Match Percentage: {close_match_percentage:.2f}%\n")

    print(f"Weighted Accuracy: {weighted_accuracy:.4f}")


# --------------------------------------------------------------


if __name__ == "__main__":

    # loading movielens dataset
    movies_df, users_df, rating_df, rating_test_df = load_data()

    # create a Lusifer object
    lusifer = Lusifer(users_df=users_df,
                      items_df=movies_df,
                      ratings_df=rating_df)

    # set API connection
    # model = "gpt-4o-mini-2024-07-18"
    model = "gpt-4o-mini"
    lusifer.set_openai_connection(KEY, model=model)

    # set column names
    lusifer.set_column_names(user_feature="user_info",
                             item_feature="movie_info",
                             user_id="user_id",  # set by default
                             item_id="movie_id",
                             timestamp="timestamp",  # set by default
                             rating="rating")  # set by default

    # set LLM initial instruction
    instructions = """You are an AI assistant that receives users information and try to act like the user  by 
    analysing user's characteristics, profile and historical ratings inorder to provide new ratings for the recommended movies"""

    lusifer.set_llm_instruction(instructions)

    # you can set the prompts as below, or ignor this and use the default prompts
    lusifer.set_prompts()

    # you can set the path to store intermediate storing procedure. By default, they will be saved on Root.
    # lusifer.set_saving_path(self, path="")

    # Filtering out invalid movie_ids, make sure we have movie_info for every movie in the test set
    rating_test_df = lusifer.filter_ratings(rating_test_df)
    rating_df = lusifer.filter_ratings(rating_df)

    # users_df = lusifer.filter_users()


    user_ids = rating_test_df['user_id'].unique()
    # user_ids = [10]

    temp_token_counter = 0
    previous_token_total = 0

    ## PHASE 1: Generating Summary of User's Behavior

    # Generating user profile
    for user_id in tqdm(user_ids, desc="Processing users and generating summary profile"):
        if users_df.loc[users_df['user_id'] == user_id, 'user_profile'].any():
            continue

        else:
            # generating summary for the user using Lusifer
            user_profile = lusifer.generate_user_profile(user_id, recent_items_to_consider=40)
            users_df.loc[users_df['user_id'] == user_id, 'user_profile'] = user_profile

            # Saving summaries
            lusifer.save_data(users_df, 'users_with_summary_df')

    ## PHASE 2: Generating Simulated ratings
    for user_id in tqdm(user_ids, desc="Generating simulated ratings"):
        # isolating user's ratings in the test set
        user_ratings = rating_test_df[rating_test_df['user_id'] == user_id]

        # we might have some values from previous run
        missing_ratings = user_ratings[user_ratings['simulated_ratings'].isnull()]

        # getting the summary for the user
        user_profile = users_df.loc[users_df['user_id'] == user_id, 'user_profile'].values[0]

        # we will not run this part if we have all ratings for the user
        if not missing_ratings.empty:
            last_N_movies = lusifer.get_last_ratings(user_id, n=10)

            # generate rating
            llm_ratings = lusifer.rate_new_items(user_profile, last_N_movies, missing_ratings)

            # Assigning the ratings to the movies
            for movie_id, rating in llm_ratings.items():
                rating_test_df.loc[(rating_test_df['user_id'] == user_id) & (
                        rating_test_df['movie_id'] == movie_id), 'simulated_ratings'] = rating

            lusifer.save_data(rating_test_df, 'rating_test_df_test')

    total_prompt_tokens = lusifer.total_tokens['prompt_tokens'] + lusifer.total_tokens['completion_tokens']
    total_cost = ((lusifer.total_tokens['prompt_tokens'] / 1000000) * 0.15) + (
            (lusifer.total_tokens['completion_tokens'] / 1000000) * 0.6)  # Cost calculation estimation

    print("\nToken Usage and Cost:")
    print(f"Prompt Tokens: {lusifer.total_tokens['prompt_tokens']}")
    print(f"Completion Tokens: {lusifer.total_tokens['completion_tokens']}")
    print(f"Total Tokens: {lusifer.total_tokens['prompt_tokens'] + lusifer.total_tokens['completion_tokens']}")
    print(f"Estimated Cost (USD): {total_cost:.5f}")

    print("\nOverall Metrics:")

    evaluate_result(rating_test_df)

