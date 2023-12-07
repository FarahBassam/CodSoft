import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(18,8)},style='darkgrid')
from time import time
import re
import string
import nltk
from googletrans import Translator
from langdetect import detect
import pycountry
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import *
import warnings
warnings.filterwarnings('ignore')
#read training dataset and preview the output in 3 columns
trainedSet = pd.read_csv(r"C:\Users\wo-20\Downloads\train_data.txt",sep=':::',names=['Title', 'Genre', 'Description'],engine='python').reset_index(drop=True)
trainedSet.head() #preview first rows of the dataset
#read the test dataset and also preview the output in 3 columns:title, genre, Description
testSet = pd.read_csv(r"C:\Users\wo-20\Downloads\test_data.txt",sep=':::',names=['Title', 'Genre', 'Description'],engine='python').reset_index(drop=True)
testSet.head()  #preview first rows of the dataset
trainedSet.describe(include='object')#analyze the train data
trainedSet.info() #check for null values
# Check for duplicates and sum them
duplicate_sum = trainedSet.duplicated().sum()
# Display the sum of duplicates
print("Sum of duplicate rows:", duplicate_sum)
trainedSet.Genre.unique() #No anomalies values
testSet.describe(include='object')#analyze the test data
testSet.info()
# Check for duplicates and sum them
duplicate_sum = testSet.duplicated().sum()
# Display the sum of duplicates
print("Sum of duplicate rows:", duplicate_sum)
#Now we need to detect if there is different langs other than english
lang = trainedSet['Description'].apply(lambda x: detect(str(x)) if not pd.isnull(x) else 'unknown')
print("Unique languages in the dataset:", lang.unique())
def langdetect(text):
    try:
        return pycountry.languages.get(alpha_2=detect(text)).name.lower()
    except:
        return 'Unknown'def datacleaning(text):
    text = re.sub(f'[{string.digits}]','',text) #remove digits
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text) #remove chars
    text = re.sub(f'[{string.punctuation}]','',text) #remove punctations
    text = re.sub('Mail <svaradi@sprynet.com> for translation. ','',text) #remove mails
    text = re.sub(r'@\S+', '', text) #remove mentions
    text = re.sub(r'http\S+', '', text) #remove http links
    return text
#Apply functuin on both test and train datasets
trainedSet['Description'] = trainedSet['Description'].apply(datacleaning)
testSet['Description'] = testSet['Description'].apply(datacleaning)
trainedSet['Language'] = trainedSet['Description'].apply(langdetect)
testSet['Language'] = testSet['Description'].apply(langdetect)
trainedSet.head()
countlang= trainedSet.Language.value_counts()
fig,axs = plt.subplots(1,2)
sns.countplot(data=trainedSet,y='Language',order=countlang.index.tolist(),ax=axs[1],color='purple')
axs[1].bar_label(axs[1].containers[0])
axs[0].pie(countlang.values.tolist(),autopct='%.2f%%')
#Translate other languages
def trans(text):
    try:
        return Translator().translate(text,dest='en').text
    except:
        return text
trainedSet.loc[~trainedSet['Language'].isin(['english']), 'Description']=trainedSet.loc[~trainedSet['Language'].isin(['english']),'Description'].apply(trans)
testSet.loc[~testSet['Language'].isin(['english']), 'Description']=testSet.loc[~testSet['Language'].isin(['english']),'Description'].apply(trans)
#check the distribution of genres
myaxis=sns.countplot(data=trainedSet,x='Genre',order=trainedSet.Genre.value_counts().index,palette='viridis')
myaxis.bar_label(myaxis.containers[0])
plt.title('Genres Distribution')
plt.xticks(rotation=45)
plt.show()
# Using TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(lowercase=True, #Lowercase chars
                                   ngram_range=(1,1), #Capture only single words in each text(unigrams)
                                   stop_words='english',#Remove stop_words
                                   min_df=2)#Ignore words that appears less than 2 times
trainx = tfidf_vectorizer.fit_transform(trainedSet['Description'])
testx = tfidf_vectorizer.transform(testSet['Description'])
#since drama and documentary have the majority of our data,
#we should avoid imbalance data in our model by randomoversampling
sampler = RandomOverSampler()
#pass to it the output of TfidfVectorizer from train data
trainx_resampled , trainy_resampled = sampler.fit_resample(trainx,trainedSet['Genre'])
#recheck the distribution of genres
myaxis=sns.countplot(data=trainy_resampled,x=trainy_resampled.values,palette='viridis')
myaxis.bar_label(myaxis.containers[0])
plt.title('Genres Distribution')
plt.xticks(rotation=45)
plt.show()
#Check for length of our data 
print('Trained Set:',trainx_resampled.shape[0])
print('Test Set:',trainy_resampled.shape[0])
#Get the actual solutions to compare it with our predictions
actualSol = pd.read_csv(r"C:\Users\wo-20\Downloads\test_data_solution.txt",engine="python",
                      sep=':::',usecols=[2],header=None).rename(columns={2:'Actual Genre'})
actualSol.head()
#Naive Bayes Model
Naive = MultinomialNB(alpha=0.3)
starting = time()
Naive.fit(trainx_resampled,trainy_resampled)
predictedY = Naive.predict(testx)
print('Accuracy:',accuracy_score(actualSol,predictedY))
ending= time()
print('Running Time: ',round(ending - starting,2),'secounds')
report = classification_report(actualSol, predictedY)
print(report)
confusionMatrix =confusion_matrix(actualSol,predictedY,labels=Naive.classes_)
confusionMatrixD = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix,display_labels=Naive.classes_)
confusionMatrixD.plot(cmap=plt.cm.Reds,xticks_rotation='vertical',text_kw={'size': 7})
plt.show()
# Creating a Pandas Series from the predicted labels
predicted_series = pd.Series(predictedY)

# Concatenating the testSet DataFrame and the actual Sol DataFrame along columns
concatenated_df = pd.concat([testSet, actualSol], axis=1)

# Concatenating the concatenated DataFrame and 'predicted_series' along columns
result_df = pd.concat([concatenated_df, predicted_series], axis=1)

# Renaming the column containing predicted labels as 'Predicted_Genre'
result_df = result_df.rename(columns={0: 'Predicted_Genre'})

# Displaying the first 10 rows of the resulting DataFrame
result_df.head(20)
#try to increase the accuracy by categorizing genres other than documentrary and drama as single category
#in order to balance the data
trainYmodified = trainedSet['Genre'].apply(lambda genre: genre if genre.strip() in ['drama','documentary'] else 'other')
actualSolmodified = actualSol['Actual Genre'].apply(lambda genre: genre if genre.strip() in ['drama','documentary'] else 'other')
NaiveB = MultinomialNB(alpha=0.3)
starting = time()
NaiveB.fit(trainx,trainYmodified)
predictedY = NaiveB.predict(testx)
print('Accuracy :',accuracy_score(actualSolmodified,predictedY))
ending = time()
print('Running Time : ',round(ending - starting,2),'secounds')
# Creating a Pandas Series from the predicted labels
predicted_series = pd.Series(predictedY)

# Concatenating the testSet DataFrame and the actual Sol DataFrame along columns
concatenated_df = pd.concat([testSet, actualSolmodified], axis=1)

# Concatenating the concatenated DataFrame and 'predicted_series' along columns
result_df = pd.concat([concatenated_df, predicted_series], axis=1)

# Renaming the column containing predicted labels as 'Predicted_Genre'
result_df = result_df.rename(columns={0: 'Predicted Genre'})

# Displaying the first 10 rows of the resulting DataFrame
result_df.head(20)

axs[0].legend(labels=countlang.index.tolist(),loc='lower right')
fig.show()
#detect for symbols such as mails @
trainedSet.loc[trainedSet['Description'].str.contains(r'@\S+')].shape[0]
trainedSet.loc[trainedSet['Description'].str.contains(r'@\S+')].head()
#detect for symbols such as http links 
trainedSet.loc[trainedSet['Description'].str.contains(r'http\S+')].shape[0]
trainedSet.loc[trainedSet['Description'].str.contains(r'http\S+')].head()
trainedSet.loc[trainedSet['Description'].str.contains(r'http\S+')]['Description'].iloc[0]
