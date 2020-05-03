# AMAZON-MUSICAL-INSTRUMENTS-REVIEWS-ANALYSIS-PREDICTION
# AMAZON REVIEWS ANALYSIS &amp; PREDICTION USING PIPELINE,ML MODEL.


###### INTRODUCTION ############################################

###### AMAZON MUSICAL INSTRUMENTS REVIEWS

# Data Cleaning Libraries
import numpy as np
import pandas as pd

# Data Visulation Libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as py

# Data Prediction Libraries
from pygments.lexers import go
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report

# Data set (Amazon Musical Instruents Review)
AMIR = pd.read_csv(r'C:\\Users\\Siddharth\\Desktop\\Musical_instruments_reviews.csv',engine='python',encoding='utf-8')

print(AMIR)

# AMIR dataset info.
print(AMIR.info())

# Dropping null rows
AMIR = AMIR.dropna()
print(AMIR.info())

# AMIR Dataset
print(AMIR.head())

# Length of words in each message of review text column
AMIR['Length of Words'] = AMIR['reviewText'].apply(lambda x : len(x.split()))

# renaming 'overall' column with 'rating'
AMIR.rename(columns={'overall':'rating'},inplace=True)
print(AMIR.head(4))

# Total Number of Users who rated the product as per rating category
print(AMIR.groupby(by='rating').helpful.count())

# Overall Rating with respect to Length of words in reviewtext messages
g = sns.FacetGrid(AMIR,col='rating')
g.map(sns.kdeplot,'Length of Words',color='red')
plt.show()

# Total number of people that rated the products as per rating category
import plotly.graph_objects as go

paris = go.Figure(data=[go.Pie(values=AMIR.groupby(by='rating').helpful.count(),labels=[1,2,3,4,5],
                       title='Volume received by each rating category.')])
paris.show()

# Predicting whether the reviewText message is positive or negative
# Considering Rating '1,2,3' as 'Negative Review'
# Considering Rating '4,5' as 'Positive Review'

review = {1:'Negative',2:'Negative',3:'Negative',4:'Positive',5:'Positive'}
AMIR['review'] = AMIR['rating'].map(review)
AMIR[['reviewText','rating','review']].head()
print(AMIR[['reviewText','rating','review']].head(25))
print("\n")

# Selecting Features & Labels
X = AMIR['reviewText']        # features
y = AMIR['review']            # labels

# Splitting data into Training Data & Test Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Pipeline
pipeline = Pipeline([
    ('Count Vectorizer',CountVectorizer()),
    ('Model',MultinomialNB())
])

# Training Data
pipeline.fit(X_train,y_train)

# Model Prediction
y_pred = pipeline.predict(X_test)
print(y_pred)
print("\n")

# Model Evaluation
print(confusion_matrix(y_test,y_pred))
print('\n')
print(classification_report(y_test,y_pred))


