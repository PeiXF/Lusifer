
import openai
from openai import OpenAI
import json
import re
import time
import pandas as pd
import logging
from local_llm.LocalLM import LocalLM


class Lusifer:
    """
    LLM-based User SImulated Feedback Environment for online Recommender systems:
    Lusifer can generate user summary behavior, updates the summary, and generate simulated ratings
    """

    def __init__(self, users_df, items_df, ratings_df):
        # loading data as pandas dataframes
        self.users_df = users_df
        self.items_df = items_df
        self.ratings_df = ratings_df

        self.api_key = None  # will be set by user
        self.model = None  # will be set by user

        self.user_feature = None  # will be set by user
        self.user_id = None
        self.item_feature = None  # will be set by user
        self.item_id = None
        self.timestamp = None
        self.rating = None

        # to trace the number of tokens and estimate the cost if needed
        self.temp_token_counter = 0
        self.total_tokens = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
        }
        self.RPD = 0
        self.token_clock = 0

        # prompts
        self.instructions = None
        self.prompt_user_profile = None
        self.prompt_update_user_profile = None
        self.prompt_simulate_rating = None

        # saving path
        self.saving_path = ""
        # self.llm = LocalLM(model="gemma3:4b")
        self.llm = LocalLM(model="gpt-3.5-turbo-16k")

    # --------------------------------------------------------------
    def set_openai_connection(self, api_key, model):
        """
        Setting openai connection
        :param api_key: openai Key
        :param model: LLM model from openai API
        :return:
        """
        self.api_key = api_key
        self.model = model

    # --------------------------------------------------------------
    def set_column_names(self, user_feature, item_feature,
                         user_id="user_id",
                         item_id="item_id",
                         timestamp="timestamp",
                         rating="rating"):
        """
        Setting necessary column names
        :param user_feature: user feature column
        :param item_feature: item feature column
        :param user_id: user_id column
        :param item_id: item_id column
        :param timestamp: timestamp column
        :param rating: rating column
        :return:
        """

        self.user_feature = user_feature  # will be set by user
        self.item_feature = item_feature  # will be set by user

        self.user_id = user_id
        self.item_id = item_id

        self.timestamp = timestamp
        self.rating = rating

    # --------------------------------------------------------------
    def set_llm_instruction(self, instructions):
        """
        Set initial instruction of the LLM model
        :param instructions:
        :return:
        """

        self.instructions = instructions

    # --------------------------------------------------------------
    def set_prompts(self, prompt_user_profile=None, prompt_update_user_profile=None, prompt_simulate_rating=None, instructions=None):
        """
        Set prompts for Lusifer
        :param prompt_user_profile: prompt to generate the first summary
        :param prompt_update_user_profile: prompt to update the summary
        :param prompt_simulate_rating:
        :return:
        """
        if prompt_user_profile is None:
            self.prompt_user_profile = """
    You are an expert data analyst specializing in user behavior and preferences in movies.

    **Task**: Analyze the user's characteristics based on their previous ratings and provide an in-depth profile.
    
    **Instructions**:
    - Identify genres, themes, directors, actors, and other movie attributes the user enjoys or dislikes.
    - Describe the qualities or characteristics of movies that the user enjoys (based on high ratings like 4 or 5) and those they dislike (based on low ratings, like 1 or 2) without mentioning movie titles or ratings.
    - Highlight any patterns or preferences evident from the ratings.
    - The profile should be detailed enough to predict the user's potential ratings for unseen movies.
    - Ensure the profile is comprehensive and presents a cohesive analysis of the user's preferences.
    - Avoid mentioning the process or using phrases like "the user seems to".
    """
        else:
            self.prompt_user_profile = prompt_user_profile


        if prompt_update_user_profile is None:
            self.prompt_update_user_profile = """
    You are updating an existing user profile based on new rating data.
    
    **Task**: Integrate the new ratings into the user's profile to refine and enhance the analysis.
    
    **Instructions**:
    - Incorporate insights from the new ratings with the existing profile by mentioning shifts in tastes or new patterns revealed by the latest ratings.
    - Update any changes in user profiles or new patterns and preferences that emerge or rise to make the user-profile richer.
    - Ensure the updated profile is comprehensive and presents a cohesive analysis of the user's preferences.
    - Do not reference previous summaries or indicate that this profile has been updated.
    - Describe the qualities or characteristics of movies that the user enjoys (based on high ratings like 4 or 5) and those they dislike (based on low ratings, like 1 or 2).
     """
        else:
            self.prompt_update_user_profile = prompt_update_user_profile


        if prompt_simulate_rating is None:
            self.prompt_simulate_rating = """
    You are an expert movie critic and data analyst.

    **Task**: Based on the user's profile and recent movie interactions, rate the following movies as YOU ARE the user.
    
    **Instructions**: 
    - Carefully consider the user's preferences, dislikes, and behavioral patterns as outlined in the user profile. 
    - By analyze each movie's attributes (genre, director, actors, summary) and determine how well 
    it aligns with the user's tastes, assign a rating from 1 to 5 for each movie, where 1 is the lowest (strong 
    dislike) and 5 is the highest (strong like). 
    - Do not include any explanations or additional text; only provide the ratings in the specified format.
    
    **Movies to Rate**:
    - Each movie is provided with its ID and description.
    """
        else:
            self.prompt_simulate_rating = prompt_simulate_rating

        if instructions is None:
            self.instructions =  """
    You are an advanced language model assistant designed to perform specific tasks based on instructions.
    
    **General Guidelines**:
    - Follow the prompt instructions precisely.
    - Provide clear, concise, and accurate information.
    - Ensure all outputs are properly formatted as per the instructions.
    
    """
        else:
            self.instructions = instructions

    # --------------------------------------------------------------

    def set_saving_path(self, path=""):
        """
        Setting openai connection
        :param path: path to the folder you want to store the intermediate progress of Lusifer
        :return:
        """
        self.saving_path = path

    # --------------------------------------------------------------
    def get_last_ratings(self, user_id, n=20):
        """
        Retrieve last N ratings according to the timestamp
        :param user_id:
        :param n: int or None (default is 20)
        :return: DataFrame
        """
        user_ratings = self.ratings_df[self.ratings_df[self.user_id] == user_id].sort_values(by=self.timestamp,
                                                                                             ascending=False)
        if n is not None:
            user_ratings = user_ratings.head(n)
        user_items = self.items_df[self.items_df[self.item_id].isin(user_ratings[self.item_id])]
        return user_ratings.merge(user_items, on=self.item_id)

    # --------------------------------------------------------------
    def generate_user_profile_prompt(self, user_info, last_n_items, update=False, current_profile=None):
        """
        Generating the initial prompt to capture user's characteristics
        :param user_info: user information
        :param last_n_items: dataframe containing last n items
        :param update: True if we want to update the user_profile
        :param current_profile: current user_profile
        :return:
        """
        if update:
            prompt_user_profile = self.prompt_update_user_profile
            user_info = current_profile
        else:
            prompt_user_profile = self.prompt_user_profile


        # getting rating summary and movie summaries: Below is the sample based on Movielens data
        ratings_summary = '\n'.join(
            f"- **Movie**: {row[self.item_feature]} \n **Rating**: {row[self.rating]} \n"
            for _, row in last_n_items.iterrows()
        )

        # Generating prompt
        prompt = f"""You are provided with the following information about a user:

    **User Profile**: 
    {user_info}
    
    **Recent Ratings**::
    {ratings_summary}
        
    {prompt_user_profile}
    
    {self.user_profile_output_prompt(update)}
    
    """

        return prompt

    # --------------------------------------------------------------
    def user_profile_output_prompt(self, update):
        """
        output instructions
        :return: prompt
        """

        if update:

            output_instructions = """
            **Output Format**:
            - Provide the user_profile in a JSON object under the key "user_profile", where the value is a cohesion and coherence structured text describing the user's profile in two paragraphs.
            - In the JSON object under the key "update", provide brief highlight of updates and changes in user_profile comparing to the previous profile (in one sentence).
            - Ensure the JSON is properly formatted for parsing, with no other keys other than "user_profile".
            - Do not include any additional text outside the JSON object.
            
            {
              "user_profile": "Detailed analysis of the user's movie preferences as an structured plain text"
              "update": "Short and brief changes in the user_profile comparing to the previous profile after new ratings"
            }

            **Example output structure**:
            {
              "user_profile": "User is interested in comedy and dramatic movies."
              "update": "User is showing new interest to dramatic movies"
            }

            """

        else:

            output_instructions = """
            **Output Format**:
            - Provide the user_profile in a JSON object under the key "user_profile", where the value is a cohesion and coherence structured text describing the user's profile.
            - Ensure the JSON is properly formatted for parsing, with no other keys other than "user_profile".
            - Do not include any additional text outside the JSON object.
                    
            {
              "user_profile": "Detailed analysis of the user's movie preferences as an structured plain text"
            }
            
            **Example output structure**:
            {
              "user_profile": "User is interested in comedy and dramatic movies."
            }
            
            """

        return output_instructions

    def simulated_rating_output_prompt(self, test_set):
        """
        instructions for the output format in simulating the ratings process
        :return:
        """
        lines = [
            f'"{row[self.item_id]}": "simulated_rating (int)"'
            for _, row in test_set.iterrows()
        ]
        output_example = "{\n" + ",\n".join(lines) + "\n}"


        output_instructions = """
        
        **Output Requirements**:
        - Provide the ratings as JSON output where the keys are movie IDs and ratings are values. 
        - ratings should be integers scale 1 to 5.
        - Ensure the JSON is properly formatted for parsing, with no other keys other than movie_ids.
        - Do not include any additional text outside the JSON output.
        
        **Example of ACCEPTED output structure:** \n
             
        """

        output_instructions += output_example

        return output_instructions

    # --------------------------------------------------------------
    def rate_new_items_prompt(self, user_profile, last_n_movies, test_set):
        """
        Generate the proper prompt to ask the LLM to provide ratings for the recommendations
        :param user_info: user information (text)
        :param analysis: LLM's summary of user's behavior based on user's background (text)
        :param test_movies: testset
        :return:
        """

        # Recent items summaries
        recent_items_summary = '\n'.join(
            f"- **Movie**: {row[self.item_feature]}, **Rating**: {row[self.rating]}"
            for _, row in last_n_movies.iterrows()
        )

        # Test items summaries
        items_summary = '\n'.join(
            f"- **Movie ID**: {row[self.item_id]}\n  **Description**: {row[self.item_feature]}"
            for _, row in test_set.iterrows()
        )

        prompt = f"""You are provided with the following information about a user:

        **User Profile**:
        {user_profile}

        **User's Recent Ratings**:
        {recent_items_summary}

        {self.prompt_simulate_rating}

        {items_summary}
        
        {self.simulated_rating_output_prompt(test_set)}
        """

        return prompt

    # --------------------------------------------------------------
    def generate_user_profile(self, user_id, recent_items_to_consider=40, chunk_size=10):
        """
        Generates user's summary of behavior
        :param user_id: int
        :param recent_items_to_consider:
        :param chunk_size: int
        :return:
        """

        # Retrieve user information
        user_info = self.users_df[self.users_df['user_id'] == user_id][self.user_feature].values[0]
        last_n_items = self.get_last_ratings(user_id, n=recent_items_to_consider)

        # Initialize the user profile and a dictionary to store intermediate profiles
        updated_user_profile = None
        intermediate_profiles = {}
        profile_changes = {}

        # Process ratings in chunks
        for chunk_number, i in enumerate(range(0, len(last_n_items), chunk_size), start=1):
            chunk = last_n_items[i:i + chunk_size]
            if not chunk.empty:
                if updated_user_profile is None:
                    # Generate initial user profile
                    prompt = self.generate_user_profile_prompt(user_info=user_info, last_n_items=chunk)
                    # Get response from LLM and update the profile
                    updated_user_profile, tokens_chunk = self.get_llm_response(prompt, mode="user_profile")
                    update = "First User profile"
                else:
                    # Update existing user profile
                    prompt = self.generate_user_profile_prompt(
                        user_info=user_info,
                        last_n_items=chunk,
                        update=True,
                        current_profile=updated_user_profile
                    )

                    # Get response from LLM and update the profile
                    updated_user_profile, update, tokens_chunk = self.get_llm_response(prompt, mode="user_profile", update=True)

                # Update token counters
                self.update_limit_tracker(tokens_chunk)

                # Store the intermediate profile with the chunk number as the key
                intermediate_profiles[f'chunk_{chunk_number}'] = updated_user_profile
                profile_changes[f'chunk_{chunk_number}'] = update

        # Update the final user profile in the DataFrame
        self.users_df.loc[self.users_df['user_id'] == user_id, 'user_profile'] = updated_user_profile

        # Store the intermediate profiles in a new column for easy access and evaluation
        self.users_df.loc[self.users_df['user_id'] == user_id, 'intermediate_user_profiles'] = [intermediate_profiles]
        self.users_df.loc[self.users_df['user_id'] == user_id, 'profile_changes'] = [profile_changes]

        return updated_user_profile

    # --------------------------------------------------------------
    def update_limit_tracker(self, tokens):
        """
        updates the number of tokens and requested used
        :param token:
        :return:
        """
        # Update token counters
        self.total_tokens['prompt_tokens'] += tokens['prompt_tokens']
        self.total_tokens['completion_tokens'] += tokens['completion_tokens']

        self.temp_token_counter += tokens['prompt_tokens']

        if self.temp_token_counter > 195000:  # Using a safe margin
            print("Sleeping to respect the token limit...")
            # reset the token counter
            self.temp_token_counter = 0
            time.sleep(30)  # Sleep for a minute before making new requests

        self.RPD = + 1

        if self.RPD == 10000:
            print("You have sent 10,000 requests ... you cannot send any more requests for today")


    # --------------------------------------------------------------
    def rate_new_items(self, user_profile, last_n_items, test_set, test_chunk_size=1):
        """
        Generate final output: simulated ratings for the entire test set
        """

        test_items = test_set.merge(self.items_df, on=self.item_id)
        recent_items = last_n_items

        # Breaking test_set into chunks
        test_item_chunks = [test_items[i:i + test_chunk_size] for i in range(0, len(test_items), test_chunk_size)]

        aggregated_ratings = {}

        for chunk in test_item_chunks:
            prompt = self.rate_new_items_prompt(user_profile, recent_items, chunk)
            # response, tokens = self.get_llm_response(prompt, mode="rating")
            llm_response, tokens = self.llm.get_llm_response(prompt, mode="rating")

            # Parse the LLM output
            cleaned_ratings = self.parse_llm_ratings(llm_response)

            aggregated_ratings.update(cleaned_ratings)

            self.update_limit_tracker(tokens)

        return aggregated_ratings

    # --------------------------------------------------------------
    def get_llm_response(self, prompt, mode, update=False, max_retries=3):
        """
        sending the prompt to the LLM and get back the response
        """

        openai.api_key = self.api_key
        instructions = self.instructions

        client = OpenAI(api_key=self.api_key)

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": instructions},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=700,
                    n=1,
                    temperature=0.7
                )

                tokens = {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }


                try:
                    output = json.loads(response.choices[0].message.content)

                    if mode == "user_profile":
                        if 'user_profile' not in output:
                            print(f"'user_profile' is missing in response on attempt {attempt + 1}. Retrying...")
                            continue  # Continue to next attempt

                        else:
                            if update:
                                if 'update' not in output:
                                    print(
                                        f"'update' is missing in response on attempt {attempt + 1}. Retrying...")
                                    continue  # Continue to next attempt
                                else:
                                    return output["user_profile"], output["update"], tokens
                            else:
                                return output["user_profile"], tokens


                    elif mode == "rating":
                        # Check if all keys and values are integers or strings that can be converted to integers
                        all_int = True
                        for k, v in output.items():
                            try:
                                int(k)
                                int(v)
                            except ValueError:
                                all_int = False
                                break
                        if all_int:
                            return output, tokens
                        # make sure key and values are integers if they are:
                        else:
                            print(f"Keys and values are not integers on attempt {attempt + 1}. Retrying...")
                            continue  # Continue to next attempt
                    else:
                        print(f"Invalid mode: {mode}")
                        return None, tokens

                except json.JSONDecodeError:
                    print(f"Invalid JSON from LLM on attempt {attempt + 1}. Retrying...")

            except openai.APIConnectionError as e:
                print("The server could not be reached")
                print(e.__cause__)  # an underlying Exception, likely raised within httpx.

            except openai.RateLimitError as e:
                print("A 429 status code was received; we should back off a bit.")
                backoff_time = (2 ** attempt)
                time.sleep(backoff_time)
            except openai.APIStatusError as e:
                print("Another non-200-range status code was received")
                print(e.status_code)
                print(e.response)

        print("Max retries exceeded. Returning empty response.")
        return [], {}

    # --------------------------------------------------------------
    def save_data(self, df, file_name):
        """
        Save the updated dataframes to files after each update to have checkpoint
         :return:
        """
        # Save the updated dataframes to files
        df.to_pickle(f'{self.saving_path}{file_name}.pkl')
        df.to_csv(f'{self.saving_path}{file_name}.csv')

    # --------------------------------------------------------------
    def filter_ratings(self, rating_test_df):
        """
        Make sure we have information about all the items in the test set
        :param rating_test_df: dataframe
        :return:
        """
        valid_item_ids = self.items_df[self.item_id].unique()
        self.ratings_df = self.ratings_df[self.ratings_df[self.item_id].isin(valid_item_ids)]
        rating_test_df = rating_test_df[rating_test_df['movie_id'].isin(valid_item_ids)]
        return rating_test_df

    def filter_test_ratings(self, rating_test_df, test_case=10):
        """
        Make sure we have information about all the items in the test set
        :param rating_test_df: dataframe
        :return:
        """
        # 3. Group by user_id and take only the first 'test_case' rows for each user
        rating_test_df = (
            rating_test_df
            .groupby('user_id', group_keys=False)
            .apply(lambda group: group.head(test_case))
        )
        return rating_test_df

    # --------------------------------------------------------------
    def filter_users(self):
        """
        Make sure we have information about all the items in the test set
        :param rating_test_df: dataframe
        :return:
        """
        self.users_df = self.users_df[self.users_df['user_id'].isin(self.ratings_df)]
        return self.users_df

    # --------------------------------------------------------------
    def clean_key(self, key):
        """
        Handles unexpected LLM outputs: parsing key (item ids)
        :param key:
        :return:
        """
        # Use regex to extract numeric part of the key
        match = re.search(r'\d+', key)
        if match:
            return int(match.group(0))
        return None

    # --------------------------------------------------------------
    def clean_value(self, value):
        """
        Handles unexpected LLM outputs: parsing value (simulated rating)
        :param value:
        :return:
        """
        # Attempt to convert the value to an integer
        try:
            return int(value)
        except ValueError:
            return None

    # --------------------------------------------------------------
    def parse_llm_ratings(self, llm_ratings):
        """
        Parsing LLM output and handling common unexpected cases
        :param llm_ratings: dictionary of simualted ratings {item_id: rating}
        :return:
        """
        cleaned_ratings = {}
        if len(llm_ratings) == 0:
            return None
        else:
            for movie_id, rating in llm_ratings.items():
                clean_item_id = self.clean_key(movie_id)
                clean_rating = self.clean_value(rating)
                if clean_item_id is not None and clean_rating is not None:
                    cleaned_ratings[clean_item_id] = clean_rating
            return cleaned_ratings

    # --------------------------------------------------------------
    # Generate simulated rating for a single user and item
    def simulate_rating(self, user_id, recommended_items_list):
        """
        Generate simulated rating for a single user and a single recommended item based on user_profile
        and record the new interaction in self.ratings_df
        """
        # Get user_profile
        user_profile_series = self.users_df[self.users_df[self.user_id] == user_id]['user_profile']
        if user_profile_series.empty:
            print(f"User ID {user_id} not found in users_df")
            return None
        user_profile = user_profile_series.values[0]

        # Get last N items rated by the user for context (e.g., last 10)
        last_n_items = self.get_last_ratings(user_id, n=10)
        if last_n_items.empty:
            print(f"No recent ratings found for User ID {user_id}")
            last_n_items = pd.DataFrame(columns=[self.item_feature, self.rating])

        # Get item info for recommended_items_list
        item_info = self.items_df[self.items_df[self.item_id].isin(recommended_items_list)]
        missing_items = set(recommended_items_list) - set(item_info[self.item_id].tolist())

        if missing_items:
            print(f"Item IDs {missing_items} not found in items_df")
            # Remove missing items from the list
            recommended_items_list = [item for item in recommended_items_list if item not in missing_items]
            if not recommended_items_list:
                print("No valid items to rate after removing missing items.")
                return None

        # Prepare test_set as DataFrame with the recommended items
        test_set = item_info

        # Generate the prompt to rate the recommended items
        prompt = self.rate_new_items_prompt(user_profile, last_n_items, test_set)

        # Get LLM response
        # llm_response, tokens = self.get_llm_response(prompt, mode="rating")
        # llm = LocalLM(model="gemma3:12b")
        llm = LocalLM(model="gpt-3.5-turbo-16k")
        # llm_response, tokens = llm.get_llm_response(prompt, mode="rating")
        llm_response, tokens = self.get_llm_response(prompt, mode="rating")

        # Parse the LLM output
        cleaned_ratings = self.parse_llm_ratings(llm_response)

        # Filter out ratings that are not in the recommended_items_list
        simulated_ratings = {item_id: rating for item_id, rating in cleaned_ratings.items() if
                             item_id in recommended_items_list}

        if simulated_ratings:
            # Record the new interactions in self.ratings_df
            new_interactions = []
            current_time = time.time()
            for item_id, rating in simulated_ratings.items():
                new_interaction = {
                    self.user_id: user_id,
                    self.item_id: item_id,
                    self.rating: rating,
                    self.timestamp: current_time,
                }
                new_interactions.append(new_interaction)
            self.ratings_df = pd.concat([self.ratings_df, pd.DataFrame(new_interactions)], ignore_index=True)

            # Optionally, save the updated ratings_df
            self.save_data(self.ratings_df, 'ratings_with_simulated_interactions')
        else:
            print(f"No valid ratings generated for the recommended items.")
            return None

        return simulated_ratings

    # --------------------------------------------------------------
    # Update user profile based on new interactions
    def update_user_profile(self, user_id, recent_items_to_consider=5):
        """
        Generates user's summary of behavior
        :param user_id: int
        :param recent_items_to_consider:
        :param chunk_size: int
        :return:
        """
        total_tokens = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
        }

        user_info = self.users_df[self.users_df[self.user_id] == user_id][self.user_feature].values[0]
        user_profile = self.users_df[self.users_df[self.user_id] == user_id]["user_profile"].values[0]

        last_n_items = self.get_last_ratings(user_id, n=recent_items_to_consider)

        prompt = self.generate_user_profile_prompt(user_info=user_info, last_n_items=last_n_items, update=True, current_profile=user_profile)
        new_user_profile, tokens_chunk = self.get_llm_response(prompt, mode="user_profile")

        # updating token counters
        self.total_tokens['prompt_tokens'] += tokens_chunk['prompt_tokens']
        self.total_tokens['completion_tokens'] += tokens_chunk['completion_tokens']

        self.users_df.loc[self.users_df['user_id'] == user_id, 'user_profile'] = new_user_profile

        return new_user_profile
