import pandas as pd
from jedi.api import file_name
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import nltk
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# Parameters
model_num =3 # 1:adaboost 2:knn 3:randomfrest

# For AdaBoost
dtree_max_depth = 20
n_estimators = 10

# For knn
cluster = 5  # keep it 5

# For Randomforest
number_of_trees = 10
max_depth = 100
 
def plot_cm(y_true, y_pred, figsize=(15,15)):
    cm = metrics.confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    splot=sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)
    sfig = splot.get_figure()
    sfig.savefig('RandomForest_maxdepth-100_numtree-10_CM.png')

def convert_dtype(x):
    if not x:
        return 0
    try:
        return int(float(x))
    except:
        return 0

df = pd.read_csv("Clothing_Shoes_and_Jewelry_10.csv", converters={'vote': convert_dtype, 'overall': convert_dtype})
df.reviewText = df.reviewText.fillna('')
df.summary = df.summary.fillna('')
df['FinalReviewText'] = df['reviewText'] + ' ' + df['summary']

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


def data_cleaning(row):
    row = row.lower()
    row = re.sub('[^A-Za-z]+', ' ', row)

    text_tokens = word_tokenize(row)
    stop_words = stopwords.words('english')

    list_clean_words = [word for word in text_tokens if not word in stop_words]

    lemmatize = WordNetLemmatizer()
    for i in range(len(list_clean_words)):
        list_clean_words[i] = lemmatize.lemmatize(list_clean_words[i])
    row = ' '.join(list_clean_words)
    return row


df['FinalReviewText'] = df['FinalReviewText'].apply(lambda x: data_cleaning(x))

X_train, X_test, Y_train, Y_test = train_test_split(df['FinalReviewText'], df['overall'], test_size=0.25,
                                                    random_state=30)
vectorizer = TfidfVectorizer()
tf_x_train = vectorizer.fit_transform(X_train)
tf_x_test = vectorizer.transform(X_test)
if model_num == 1:
    file_name = "AdaBoost_maxdepth-" + str(dtree_max_depth) + "_numtree-" + str(n_estimators)
    abc = AdaBoostClassifier(DecisionTreeClassifier(max_depth=dtree_max_depth), n_estimators=n_estimators,
                             learning_rate=1)
    model = abc.fit(tf_x_train, Y_train)
    y_predabc = model.predict(tf_x_test)
    report = metrics.classification_report(Y_test, y_predabc, zero_division=1)
    with open(file_name, 'w') as f:
        f.write(report)
    f.close()
    plot_cm(Y_test, y_predabc)
    print(file_name + "\nClassification Report :-\n\n", report)
elif model_num == 2:
    file_name = "KNN_cluster-" + str(cluster)
    classifier = KNeighborsClassifier(n_neighbors=cluster)
    classifier.fit(tf_x_train, Y_train)
    Y_predknn = classifier.predict(tf_x_test)
    report = metrics.classification_report(Y_test, Y_predknn, zero_division=1)
    with open(file_name, 'w') as f:
        f.write(report)
    f.close()
    print(file_name + "\nClassification Report :-\n\n", report)
elif model_num == 3:
    file_name = "Randomforest_numtree-" + str(number_of_trees) + "_maxdepth-" + str(max_depth)
    rf = RandomForestClassifier(n_estimators=number_of_trees, max_depth=max_depth)
    rf.fit(tf_x_train, Y_train)
    y_pred = rf.predict(tf_x_test)
    report = metrics.classification_report(Y_test, y_pred, zero_division=1)
    with open(file_name, 'w') as f:
        f.write(report)
    f.close()
    plot_cm(Y_test, y_pred)
    print(file_name + "\nClassification Report :-\n\n", report)
