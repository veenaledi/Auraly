## Capstone Project
# Auraly

This project has three Jupyter notebooks (Mood Model, Spotify Dataset, and Phrases) that come together to form the deployable Auraly app. 


Each notebook produces a key artifact used in deployment:

• Notebook 1 — Mood Model: Creates the XGBoost model (`auraly_xgb_model.pkl`), label map (`label_map.json`), and optional scaler. These files are used to classify songs by mood.

• Notebook 2 — Clean Spotify Dataset: Produces the cleaned CSV (`spotify_mood_dataset.csv`) that contains all songs with predicted moods. This acts as the source for playlist generation.

• Notebook 3 — Short Phrases: Generates TF-IDF files (`tfidf_vectorizer.pkl` `tfidf_phrase_matrix.pkl`, and `tfidf_phrases_lookup.csv`) to map user text inputs (e.g. 'morning focus') to moods.

# NOTEBOOK 1

## 1.Business Understanding
*__1.1 Overview__*

Music consumption today is highly personalized, with streaming platforms offering tailored recommendations based on listening habits. However, when it comes to emotional resonance, listeners still spend significant time manually curating playlists that match how they feel in the moment. The traditional approach of browsing by genre or artist fail to capture the subtle emotional layers that make a song resonate. This project will make discovering music more personalized and enjoyable for casual listeners, DJs, and streaming platform users

*__1.2 Stakeholders__:*

1. Music Listeners – benefit from effortless mood based playlist creation and more emotionally resonant music discovery.
2. Streaming Platforms – gain deeper user engagement and personalization features that differentiate their service.
3. DJs & Curators – save time curating emotionally aligned sets for events or audiences.

*__1.3 Problem Statement__*

Even though music apps offer personalized recommendations, they still don’t understand how a listener feels. People often spend too much time searching for songs that match their mood because most platforms sort music by genre or artist, not emotion. This makes it hard to find the right songs quickly, and limits how personal and meaningful the listening experience can be.

*__1.4 Business Objective__*

To establish Auraly as an intelligent mood-based music classification system that enhances emotional connection and personalization in music streaming. By automating playlist creation through acoustic mood detection, Auraly aims to improve user engagement, simplify music curation, and unlock deeper, mood-driven discovery experiences across platforms.

*__1.5 Project Objectives__*

__Main Objective__

To develop an intelligent music classification system that automatically identifies the emotional mood of songs using acoustic features, enabling more intuitive and personalized music experiences.

__Specific Objectives__

1. Enable automated mood based playlist generation - Reduce manual curation time by dynamically grouping songs based on emotional tone.
2. Support personalized music discovery - Recommend songs that align with a listener’s current mood or emotional preferences.
3. Enhance user engagement across music platforms - Improve retention and satisfaction by offering emotionally resonant listening experiences tailored to individual users.

*__1.6 Research Questions__*

1. How accurately can acoustic features be used to classify the emotional mood of a song?
2. Does personalized mood-based music discovery lead to higher user engagement on streaming platforms?
3. How can mood based classification improve the way users discover and organize music?

*__1.7 Success Criteria__*

1. Accurate Mood Classification - The system achieves a high accuracy rate in classifying songs into predefined emotional categories based on acoustic features.
2. Improved User Experience - Users will report reduced time and effort in creating mood based playlists and express higher satisfaction with music recommendations through surveys or usability testing.
3. Increased Engagement Metrics - Streaming platforms or test environments will show measurable improvements in user engagement, e.g. longer listening sessions, more playlist saves, or higher interaction rates, when Auraly is integrated.

## 2. Data Understanding

*__2.1 Imporrting libraries__*

We imported python libraries from panda, numpy, matplotlib, seaborn, colection, sklearncontrctions and others which we used to help understand and read our data.

*__2.2 Loading the data__*

Data is from Kaggle and the dataset is called `278k_labelled_uri.csv.zip` Its a music based dataset.

*__2.3 Initial Exploration And EDA__*
*__2.3.2 Dataset summary__*

The dataset contains 277938 rows and 15 columns. It had no missing or duplicated values. 14 columns were numerical and 1 text.
The features used in this dataset include 'danceability', 'energy', 'loudness', 'speechiness' 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', and 'duration (ms)'

Visualisaations were created so as to see the datatypes and label distribution. A statistical summary table was made to show the statistics of the audio features of the dataset.

A boxplot was made to show the outliers of the audio features

## 3. Data Preparation

### 3.1 Data Cleaning

The mood labels are as follows:- 0: Sad 1: Happy 2: Energetic 3. Calm

Outliers were removed using iqr to improve accuracy

Duplicated rows were dropped and only one of any duplicated rows was left.

We filtered to around 10- 20k for better analysis and perfomance.

Rows with missing values were filled with numerical columns filled with their median and categorical columns filled with their mode


### 3.2 Cleaning Summary

The mood distribution showed that 'happy' with had the most songs followed by 'sad', then 'energetic and finally 'calm'.

The remaining features with outliers are tempo, energy, valence with highest features with outliers are loudness and duration(ms)

Our final cleaned dataset had a set of 80,000 from 277,938 records. with 33, 030 outliers removed.

### 3.3 Cleaned Data

Our cleaned data was saved into a csv called 'cleaned_music_data.csv'


 ## 4. Further EDA


 ### 4.1 Feature distribution

 Histograms were ploted to show the frequency of the audio features

 The mean and median were also visualized in the histograms.
 

### 4.2 Mood Feature Analysis

Histograms of the 4 moods were visualized to show their frequency.

Frequency of the moods was as follows:

1. Sad = 23,806 tracks
2. Happy = 33,784 tracks
3. Energetic = 14,355 tracks
4. Calm = 8,055 tracks

The qualities of each mood are: 

1. Sad - Low valence, low energy\nMelancholic, somber tracks
2. Happy - High valence, moderate energy\Uplifting, positive tracks
3. Energetic - High energy, high tempo\nUpbeat, intense tracks
4. Calm - Low energy, high acousticness\nRelaxing, peaceful tracks

### 4.3 Correlation Analysis

A heatmap of the faetures correlation features was plotted.

The top five correlated features are:

1. Energy and loudness
2. loudness and energy
3. acousticness and energy
4. energy and acousticness
5. lodness acousticness

After further cleaning, histograms were plotted against the features and thier means visualized.

Our final dataset had 80,000 records with 17 columns. 15 are numerical and 2 categorical. No missing values, no duplicates and outliers dealt with.

## 5. Data Preprocessing

We loaded the new dataset into a data frame called music_df

From the value_counts of the moods there was a significant class imbalance between the moods with happy being the highest and calm being the lowest.

Columns that were unnecessary were dropped as they provided little to no significance for our purpose.

We defined our X and y variables and calculated the statistics of our remaining numerical features

Perfomed a train test with a test size of 20% and a random_state of 42. We also used stratify for scaling.

We used StandardScaler to scale 3 features namely 'loudness' 'tempo' and 'duration(min)' with outlying values

The scaler was used to fit all 3 models that is logistic Regression, XGBoost, and Random Forest.

Smote was deployed to generate new data to deal with class imbalance

Dataframe was restored and its shape checked for the training set and testing set.

## 6. Modelling

The modeling will be completed in stages, including the development of a baseline model, additional models, and model tuning, followed by a comparison of the best-performing model
                                                 
 ## 6.1 Logistic Regression Model

 ### Linear Baseline Model

 The model was instantiated  and trained and predictions made on the test set

 On evaluation the model had an accuracy of 81%.


## 6.2 Random Forest 

### Non_linear model

The model was instantiated, trained and test set predicted

Accuracy of the model was 92%

## 6.3 XGBoost

Model was also instantiated, trained and test set predicted.

Accuracy was better with 94%

Matrix confusion matrixes were plotted for the 3 models

## 6.4 Feature importance.

As XGBoost had the best accuracy, a visual was plotted to show which features scored the highest.

The most important features in the dataset are: 'instrumentalness', 'energy', 'acousticness', 'danceability' and 'speechiness'

## 6.5 Hyparameter tuning

As the best perfoming model, XGBoost was tuned to see whether it would perfom better and a matrix confusion matrix visualized.

## 6.6 Saving The Best Model

The untuned XGBoost model perfomed better than the tuned model and thus was saved using joblib for deployment.

# 7. Evaluation, Recommendation and Conclusion

## 7.1 Overview

In this section, we shall evaluate the models and process implemented to process our analyze our data. The models were asseseed for their performance and Their ability to  address their research objectives.  

Three models were used to analyze our data. Logistic regression was used as our linear base model to assess the other 2 models. Random 
Forest Tree model was our non-linear baseline ensemble model and XGBoost was our other ensemble model used

## 7.2 Evaluation

* As from the result above, XGBoost lassifier ensemble model perfomed the best out of the the other 2 models achieving an accuracy of 94.4% with high f1-scores showing its ability to predict  the targets correctly. 

* Logistic regression had the lowest perfoming model accuracy of 81.3% with low f1-scores and Random Forest model had an accuracy of 92% with high .

* After tuning the XGBoost Classifer model as the best model to have better predictions, it had a slightly lower accuracy of 93.5% than The untuned XGBoost Classsifer model which had 94.4% accuracy.

* According to the confusion matrix, The untuned XGBoost ensemble model had the lowest false positives and false negatives meaning it predicts the moods better than the other models.

* The most important features with the highest importance in the dataset are: 'instrumentalness', 'energy', 'acousticness', 'danceability', 'speechiness' 'loudness', 'valence', 'duration_min', 'tempo' and 'liveness' in descending order.

* The untuned XGBoost Classifier model will be used to deploy.

## 7.3 Recommendation

* Data Enrichment - Expanding  to different genres to target different people with different taste.

* feature engineering - to have better features to avoid misclassififcation of the moods

* Class imbalance - Collecting more data for underrepresented moods so they are predicted better.

* Continuous Learning Framework – Implement periodic retraining with new data to ensure the model remains relevant.

## 7.4 Limitations

* Mood mislabelling - Even though the models performed well there was some misclassification of moods e.g happy being misclassified as energetic.

* Class imbalance - Some moods were underrepresented like calm

* Context - Meaning of different sounds may be misclassified e.g 'am bad' can have a good meaning or its actually bad.

* Music - having over a billion tracks worldwide with different meanings can be a key limitation.


## 7.5 Conclusion

In conclusion, we were able to achieve our main objective  by being able to develop a classification model that identifies the emotional mood of a song by its acoustic features. The models performed really well with the untuned XGBoost classifier outperforming the rest of the models.

Success was achievd:

1. There was a high accuracy in identifying moods of songs using their acoustic features

2. Users are able to get recommendation of songs based on their moods or are write a phase which the model can use to classify a mood and reccomend songs

3. Auraly once intergrated will be able to have personalised user experience and better interactions with users

4. Moving Forward, getting more data on under represented moods and improving on feature engineering so as to avoid misclassification


# NOTEBOOK 2

# Auraly - Mood Based Playlist Generator

This notebook forms the core data-processing and playlist-generation pipeline for Auraly, a mood-based music recommendation application. It begins by loading and cleaning a large dataset containing Spotify and YouTube audio features, then applies a pre-trained XGBoost classifier (auraly_xgb_model.pkl), developed in the main modeling notebook, to predict the emotional mood of each track.
The model classifies songs into four moods — happy, sad, energetic, and calm. Based on key audio and perceptual features such as energy, valence, tempo, loudness, and acousticness.

After prediction, the dataset is fully normalized: column names are standardized, irrelevant metadata removed, missing values and duplicates dropped, and features aligned for consistency and deployment. The resulting clean dataset (spotify_df) serves as the foundation for generating personalized playlists. Each record links a song’s Spotify URI, artist, and essential audio features to its predicted mood label.

The notebook then extends into a playlist-generation system that allows users to input either a mood keyword (e.g., “happy”) or a short descriptive phrase (e.g., “need calm focus music” or “upbeat gym vibes”). Through lightweight natural-language keyword matching and mood inference, the system maps user input to one of the four moods, ranks tracks within that category using mood-specific scoring weights (energy, valence, tempo, loudness, etc.), and outputs a curated list of around fifteen songs.

In deployment, this logic can be integrated into a web interface, where users instantly receive Spotify playlist recommendations derived from the trained model and curated dataset.
Overall, this notebook demonstrates the complete flow from data ingestion to deployable recommendation logic, bridging machine-learning mood classification with user-friendly playlist generation for real-world applications.

Beyond individual users, producers, record labels, and local music platforms can leverage this workflow to classify and recommend their own catalogues. For example, Kenyan underground music or emerging regional artists can be automatically categorized by mood and surfaced to new audiences — making Auraly both a discovery and personalization tool for global and local music ecosystems.

## Notebook Process

* The necessary libraries were imported for analysis of this data i.e pandas, numpy, zipfile, warnings, XGBoost, joblib, json and warinings.

* Our dataset is a zip file from https://www.kaggle.com/datasets/salvatorerastelli/spotify-and-youtube. Contents were extracted and data loaded to a data frame called spotify_data

* Dataset had 20,718 records with 28 columns. It had 16 numerical columns and 28 categorical.

* checked for missing values which prooved to be present in almost all columns.

* Dropped unnecessary columns and dropped any rows with missing features. The columns became 16 numerical 10 and categorical 6. New dataframe became Spotify_df

* Missing values where rechecked and duplicated values also checked and appeared to have none.

* The trained model i.e XGBoost was loaded using joblib and the label map was also loaded using json.

* A copy of the dataset was made  and called X_spotify and the 'duration(ms)' column was changed to 'duration(min)'. The necessary features to be used are danceability', 'energy', 'loudness', 'speechiness',  'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_min'

The moods were predicted using the X_spotify features 

* Mood label count became 

mood_label
Happy        12579
Sad           4701
Energetic     2753
Calm           683

* A function was created to get the moods by song and it returned playlists from all the moods proving success in assigning moods to our songs.

# NOTEBOOK 3

# Notebook Overview:***


This notebook implements the core phrase-to-playlist recommendation pipeline for Auraly, a music recommendation application based on mood. Its purpose is to allow users to receive personalized playlists by typing a mood keyword (e.g., “happy”) or a short descriptive phrase (e.g., “need calm focus music” or “upbeat gym vibes”).

It begins with raw phrases collected from friends and potential users, which are then cleaned and preprocessed through steps such as duplicate removal, lemmatization, tokenization, removal of signs, and stop-word elimination. The resulting cleaned and tokenized phrase dataset (`phrases.csv`) maps each phrase to a corresponding mood label (`Sad`, `Happy`, `Energetic`, `Calm`). A TF-IDF vectorizer is applied to transform user-input phrases into numerical representations, which are compared against the phrase dataset using cosine similarity to get the most likely mood.

The notebook also integrates a pre-processed song dataset (`spotify_mood_dataset.csv`) containing Spotify track features and pre-predicted mood labels. This was done in the 'Tracks.ipynb' file. Once a user’s mood is gotten from their input phrase, the system filters and ranks tracks within that mood based on audio and perceptual features such as energy, valence, and danceability. Playlists are then generated, typically containing about 10 curated songs.

Special handling is included for ambiguous or unmatched phrases: if the system cannot confidently place a mood from the input (e.g., a rare or unseen keyword), it prompts the user to select from all four moods, ensuring a playlist can still be delivered.

This notebook demonstrates the complete flow from raw user input to curated playlist output, bridging natural-language mood inference with a pre-classified song dataset. It enables users, producers, and music platforms to discover and experience music personalized by emotional context, enhancing both user engagement and music discovery, especially for niche or regional artists.


## 1. Data Cleaning and Preprocessing.

* Necessary libraries were imported for analyzing i.e pandas, numpy, matplotlib, seaborn, re, collections, warnings, contractions, sklearn, pickle and nltk.

* Data had 229 rows and 3 columns which are Short phrase, mood and mood label.

* There were no duplicated rows

* A Function was defined to deal with column 'short phrase' and a new column was created called 'cleaned phrase'

* The short phrase column was dropped

* A preprocessing function was defined and a new column formed 'phrases'

## 2. Mapping the Phrases

* The phrases.csv and Spotify_mood_dataset files were loaded. The label map file was also loaded to show thw moods.
* The phrases were vectorized using tfidf
* A function was created to generate a playlist according to moods
* A another fuction was also created to generate a playlist aaccording to phrase
* The Functions were tested to see whether it would produce accurate outcomes.

# **Phrase-Based Mood Playlist Generator

### Project Collaborators
1. Neema Naledi (naledineema@gmail.com)
2. June Henia (heniajune@gmail.com)
3. Morgan Amwai (morganamwai@gmail.com)
4. Brian Kimathi (machingabrian@gmail.com)
5. Mark Muriithi (mark.muriithi@gmail.com)

### Navigating the Repository

1. Jupyter Notebook:
2. Presentation slides PDF:
3. Data Report:
4. Dataset: [https://www.kaggle.com/datasets/abdullahorzan/moodify-dataset]
5. README.md: [https://github.com/veenaledi/DS-Capstone-Project.git]
6. .gitignore: Specifies files to ignore in version control



## Prerequisites
*Getting started*
1. Fork 
- Create a fork.
2. Clone 
- Type: git clone then paste the link below
(you can clone using either *SSH key*  or the *HTTPS*)
[https://github.com/veenaledi/DS-Capstone-Project.git]

## Testing
To run the cells press ctrl+shift
You'll need to download the dataset required 
You can get the dataset from:
[]

## Technologies Used
- Python: Primary programming language
- Pandas: Data manipulation and analysis
- Matplotlib: Data visualization
- Jupyter Notebook: Development environment
- Git: Commit and push to remote repository

## Contributions
Contributions to our project, *Auraly*, are welcome! If you have any suggestions, bug fixes, or additional features you'd like to add to the dashoard, please feel free to submit a pull request or open an issue.

## Support
For questions or support, please contact:
naledineema@gmail.com, heniajune@gmail.com, morganamwai@gmail.com, mark.muriithi@gmail.com, machingabrian@gmail.com

