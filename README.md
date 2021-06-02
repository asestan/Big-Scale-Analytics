# Big-Scale-Analytics-2021-Tesla <img src="https://user-images.githubusercontent.com/61697398/114187790-b0553000-9948-11eb-8cda-df0721cac5bc.png" width="35" height="30">
# Text Analytics : Prediction Of French Sentences' CEFR Level


Tesla Group's repository for the course project "Big Scale Analytics 2021" (University of Lausanne)

This projects aims to predict the CEFR level of a sentence in a foreign language (here French) based on machine learning techniques.
## Group members 🧍🧍🧍
- Adrian Sestan
- Igor Ranisavljevic
- Hugo Macedo Candido

## The Idea (project description) 📜
To improve one’s skills in a new foreign language, it is important to read texts in that language. These text have to be at the reader’s language level. However, it is difficult to find texts that are close to someone’s knowledge level (A1 to C2). The idea is to build a model for English speakers that predicts the difficulty of a French written text. This can be then used, e.g., in a recommendation system, to recommend texts (for example, recent news articles) that are appropriate for someone’s language level. If someone is at A1 French level, it is inappropriate to present a text at B2 level, as she won’t be able to understand it. Ideally, a text should have many known words and may have a few words that are unknown so that the person can improve.

In order to learn more about the subject, we consulted various research sources including the same objectives such as :
- Word lists in Reference Level Descriptions of CEFR (Common European Framework of Reference for Languages): https://euralex.org/wp-content/themes/euralex/proceedings/Euralex%202012/pp328-335%20Marello.pdf
- Duolingo's blog on the use of AI for adapting learning content based on CEFR level: https://blog.duolingo.com/the-duolingo-cefr-checker-an-ai-tool-for-adapting-learning-content/
- Duolingo's CEFR checker (what we're aiming to, but in french): https://cefr.duolingo.com/
- English vocabulary profile (the english equivalent of what we need to build our model): https://www.englishprofile.org/wordlists/evp
- Machine Learning for learner English, April 2020, International Journal of Learner Corpus Research: https://www.researchgate.net/publication/339736712_Machine_learning_for_learner_English_A_plea_for_creating_learner_data_challenges
- Rule-based and machine learning approaches for second language sentence-level readability, June 2014,  9th Workshop on Innovative Use of NLP for Building Educational Applications: https://www.researchgate.net/publication/263057662_Rule-based_and_machine_learning_approaches_for_second_language_sentence-level_readability

## Milestones & Goals 🎯
The project is separated in three main milestones :
1. Thinking how to model the problem and gathering data (French texts, sentences, news articles etc.) and labelling them with the relevant level
2. Building a model that predicts the difficulty/level of a new text
3. Evaluating how good the model is

The final goal is to provide an API with an UI wich returns the level of a given french sentence.

## Methodology :mag:

- Gather existing informations and scientific articles on the subject :white_check_mark:
- Build and labelize a dataset containing french sentences :white_check_mark:
  - Upload our data on AutoML (Google Cloud) :white_check_mark:
  - Train models (classification model) :white_check_mark:
  - Evaluate models :white_check_mark:
  - Deal with cognates :white_check_mark:
- Try different models:
  - CamemBERT :white_check_mark:
  - Linear SVC :white_check_mark:
- Create a web service (with interface) with flask and AppEngine :white_check_mark:
- Connect the web service to our API :white_check_mark:
- Compare models and use the best model :white_check_mark:
- Improve the UI :white_check_mark:

## Data 📊
The data used for this project comes from two different sources. The first data set was created by our team based on relevant literary sources. The second, larger than the first, was imposed on us by our teachers. Each dataset contains two columns, the first one contains the sentences to be analysed and the second one represents the label of each sentence (A1, A2, B1, B2, C1, C2).

Exemple :

| Sentences | Label |
| :---         |          ---: |
| Bonjour, je m'appelle Lucia. | A1 |

### First data set

In order to create our dataset, we had to find specific ressources that are a reference in the labeling of french language. We have used several sources such as grammar exercises, french language learning websites and sources provided by teachers. After collecting **1224 sentences**, we objectively assessed their levels under different selection criteria like words, verbs, cognats, grammar and expressions difficulties.

#### Number of sentences per level :
- A1 : 221
- A2 : 209
- B1 : 192
- B2 : 203
- C1 : 197
- C2 : 202

Website sources:
- French Reading Practice Bilingual reader articles: https://french.kwiziq.com/learn/reading 
- Lingua: https://lingua.com/french/reading/
- Podcast français facile (niveau B2):  https://www.podcastfrancaisfacile.com/niveau-delf-b2
- LaPhilo (EXEMPLE DE DISSERTATION EN PHILOSOPHIE): https://la-philosophie.com/exemple-dissertation-philosophique 

Book: 
- ABC-TCF Test de Connaissance du Français pour le Québec, Bruno Mègre & Sébastien Portelli, september 2014. 

### Second data set
This dataset was downloaded from the AIcrowd project associated with our course. It is divided into two parts, train data with **4800 sentences** and test data with **1200 sentences**. The train data will allow us to train the prediction models, so it is important to have a good distribution of labels (target).

#### Labels of the train data :
- A1 : 800
- A2 : 800
- B1 : 800
- B2 : 800
- C1 : 800
- C2 : 800

## Tools 🛠
Different tools are used for this project :
- Google Colab Python Notebooks
- Google Cloud Plateform (in particular : App Engine and AutoML Natural Language)
- Jupyter Python Notebook
- Python libraries : Spacy, Torch, Torchvision, Transformers, Keras, Scikit-learn, Pickle and many others...

## Preprocessing
We use the second train data set to train the prediction models described in the previous chapter. We will focus on the preprocessing of this data set by combining the following methods:
- Punctuation: we removed all '!"#$%&()*+,./:;<=>?@\[]_{|}~'] characters from sentences.
- Digit : we have removed all digits (1-9) from sentences.
- You can find the python files [here](https://github.com/TetraFaal/Big-Scale-Analytics-2021-Tesla/tree/main/Code/Python%20Colab%20for%20AutoML).

## Tokenizer
In order to tokenise the words in each sentence of our data set, we used the appropriate spacy library for the French language ('fr_core_news_sm'). This will allow our models to better identify the links between words and labels.

Later on, when using our model named CamemBERT, we used directly the tokenizer provided by CamemBERT.

## Google Cloud AutoML
Before we start explaining our different iterations, it is important to know that we have used the train dataset from AIcrowd.
For this first model, we used AutoML Natural Language which allowed us to create and deploy custom machine learning models that analyze documents, categorize them, identify their entities or evaluate their attributes.

Youy can found the three iterations data set [here](https://github.com/TetraFaal/Big-Scale-Analytics-2021-Tesla/tree/main/Data/DataAutoML-Preprocessing).
### First Iteration

As a first try using the Google Cloud AutoML, we have simply uploaded the data into AutoML without doing any preprocessing. The tool offered by Google automatically provided a usable model using a classification method. 

With a threshold set at 0.5, we have the following results:

- The precision is of 62.71%
- The recall is of 38.54%

#### Per class score

| Label | Precision | Recall |
| :---         |     :---:      |          ---: |
| A1   | 76.06%     | 67.5%    |
| A2     | 61.36%       | 33.75%      |
| B1   | 51.61%     | 20%    |
| B2     | 59.09%       | 32.5%      |
| C1  | 52.63%     | 25%    |
| C2    | 62.69%       | 52.5%      |

#### Confusion matrix
<img src="https://user-images.githubusercontent.com/61697398/120072827-ef287c00-c095-11eb-8a30-452dea2fc0af.png" width="350" height="300">

- The accuracy is of 51.88%

### Second iteration

As the first results were not satisfying, we decided to include the cognates in our second iteration.
We've worked on Google Colab to pre-process our data (punctuation), tokenize the sentences, and put the number and which cognates in the sentences where they have been identified. We used two cognates data sets, which can be found [here](https://github.com/TetraFaal/Big-Scale-Analytics-2021-Tesla/tree/main/Data/Cognates)

With a threshold set at 0.5, we have the following results:

- The precision is of 65.69%
- The recall is of 27.92%

#### Per class score

| Label | Precision | Recall |
| :---         |     :---:      |          ---: |
| A1   | 76.06%     | 67.5%    |
| A2     | 65.52%       | 23.75%      |
| B1   | 42.31%     | 13.75%    |
| B2     | 66.67%       | 12.5%      |
| C1  | 60%     | 18.75%    |
| C2    | 65.79%       | 31.25%      |

#### Confusion matrix
<img src="https://user-images.githubusercontent.com/61697398/120073171-893cf400-c097-11eb-8380-2d780fca766b.png" width="350" height="300">

- The accuracy is of 49.79%

As a conclusion to this iteration, the way we uploaded the cognates into the dataset resulted in Google AutoML not being capable of recognizing the weight of the cognates.

### Third iteration

We realised that the models created by AutoML (Natural language) did not know how to interpret tokenised sentences and that the cognates were not presented in an optimal way. Therefore, we performed a new iteration with punctuation and digits processing. We also added weight to each cognate by duplicating them in the sentences where they are present. Thanks to this, the labelling of the sentences allowed us to know the value of the cognates. 

For example, if a 10-word sentence has 3 cognates labelled as "C2", this will mean that the identified cognates do not offer a better understanding of the sentence. In another case, if a sentence with words often labelled as "B1" with cognates being labelled as "A2", this will confirm that the identified cognates allow a better understanding of the sentence even if the other words are more difficult. 

With a threshold set at 0.6, we have the following results:

- The precision is of 68.18%
- The recall is of 28.13%

#### Per class score

| Label | Precision | Recall |
| :---         |     :---:      |          ---: |
| A1   | 68.33%     | 51.25%    |
| A2     | 65%       | 32.5%      |
| B1   | 61.11%     | 13.75%    |
| B2     | 61.11%       | 13.75%      |
| C1  | 64%     | 20%    |
| C2    | 69.77%       | 37.5%      |

#### Confusion matrix
<img src="https://user-images.githubusercontent.com/61697398/120072368-dfa83380-c093-11eb-9edb-9c4c99384c2b.PNG" width="350" height="300">

- The accuracy is of 53.13%

This iteraction is the best of AutoML iterations, we will take it for combining with other models in our API.

## CamemBERT - Jupyter Model

We there tried to use an existing model called CamemBERT and adapt it to our use case. The main advantage and purpose of this model is that it is already trained on the french language. The model was mainly used to predict sentiments on text analysis. 

More information about the model is available [here](https://camembert-model.fr/)

As Google AutoML was limited in model tunning and was not letting us dealing with cognates the desired way, we decided to try a different solution to add some weights to the cognates. First, we tried the same dataset that we used first in Google AutoML to compare the results on AICrowd, without taking cognates in account. It resulted that our CamemBERT model was more accurate with an accuracy of 54%.

We then tried to remove cognates from the sentences. The idea behind that is that an english speaker would understand any given sentence with more ease if a cognate appear, even if the word is labelled as difficult in french. Thus, removing the cognates would not affect the labelisation made by our model only based on the difficulty of a french word that is understandable by a native english speaker. The accuracy of this method was indeed higher with 56.4% and led us to the 4th place in AICrowd.

The last method that was tried is described in the Jupyter notebooks, available from this [link](https://github.com/TetraFaal/Big-Scale-Analytics-2021-Tesla/tree/main/Code/NoteBook)

As a user may want to try his own predictions with our model, we used the library [pickle](https://docs.python.org/3/library/pickle.html) to save our pre-trained Model. The link to download it is also located in the notebook folder.

## Other Models - Jupyter Model

To diversify and try other methods we used another notebook containing different models. As those were less accurate than CamemBERT and Google autoML we decided not to upload them directly on this repository and therefore just provide a short summary of the results.

Below is the summary of the tested models and their respective accuracy:

![image](https://user-images.githubusercontent.com/71492453/120068511-57209780-c081-11eb-89d7-bf50a429883d.png)

We decided to further train the linear SVC, which resulted in the following confusion matrix:

![image](https://user-images.githubusercontent.com/71492453/120068581-ae266c80-c081-11eb-84bd-9828b167b180.png)

And finally a classification report: 

![image](https://user-images.githubusercontent.com/71492453/120068607-cac2a480-c081-11eb-95de-858e0f6b97ca.png)

## Combining Models and structure of our solution
We have chosen to take the best iterations of the AutoML (Natural language) model and the CamemBERT model and combine them directly on the App Engine, by averaging the results of both prediction based on their respectives accuracies. Unfortunately, the weight of the whole project (with all dependencies) was most likely too big to be deployed on Google Cloud Plateform (going from 34MB to 3.8GB) as the deployment was always aborted by Google. We therefore only deployed the AutoML endpoint. Nevertheless, the project could work locally. 

Through our web service, the prediction of the label of the submitted sentence is done in 4 steps:

1) Preprocess the sentence according to the prediction models.
2) Use the two prediction models
3) Average the two results
4) Display the final result on our web page

### Architecture schema (as imagined)
![image](https://user-images.githubusercontent.com/61697398/120075602-49c7d500-c0a2-11eb-8aca-7deb68c30f70.png)


## Repository organisation 🗂
Here's how the repo is organized :
- Data : you'll find [here](https://github.com/TetraFaal/Big-Scale-Analytics-2021-Tesla/blob/main/Data/Dataset_Aicrowd/22fdbaea-415d-4685-8132-1916959ca359_train.csv) all of the 4800 sentences that composes our  data for the project
- Code : the python [notebooks](https://github.com/TetraFaal/Big-Scale-Analytics-2021-Tesla/tree/main/Code) containing our code
- ABC-TCF book : our [litterature](https://github.com/TetraFaal/Big-Scale-Analytics-2021-Tesla/tree/main/ABC-TCF%20book) used for the french knowledge test for Quebec.

## Web service

You can try the model that was built with Google AutoML by following [this link](https://massive-incline-305713.appspot.com/) to our web app.
Here is the [AutoML API endpoint](https://automl.googleapis.com/v1/projects/79067930854/locations/us-central1/models/TCN6320175355386658816:predict) (authentification token needed)

##Video

You can find the video [here](https://youtu.be/KrqHYU_0U04)

## Tasks and Work repartition within the team members

Each team member partcipated equally in the realisation of the project

- Adrian
  - Building Notebooks
  - Building CamemBERT model
  - Building other Models
  - ReadMe contributor
  
- Hugo
  - Google Auto ML main contributor
  - Text cleaning and preprocessing tasks
  - ReadMe contributor
  
- Igor
  - Web service builder (App Engine - Logic and UI)
  - Dealing with API's
  - Managing the project on Google Cloud Plateform (and struggling with Google's support)
  - (Tearing his hear out on app deployment)




