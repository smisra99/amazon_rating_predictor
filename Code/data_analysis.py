import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import sys

#get the csv and make it a dataframe
csv_file = sys.argv[1]
print("read ", csv_file)
df = pd.read_csv(csv_file)
print("done\n\n")

split_name = csv_file.split('.')
base_name = split_name[0]

print("Getting the Verified Graph")

#Get the Number of Reviews that are verified
plt.figure(figsize=(15,8))
sns.countplot(
    x = 'verified',
    data = df,
    order = df.verified.value_counts().index
)
plt.xlabel("Verified")
plt.title("Verified Distribution")
#Writing to file
print("Writing to file")
plt.savefig(base_name + '_Verified_Distribution_plot.png')
print("done\n\n")

print("Getting the Votes Graph")
df_without_nan = df.dropna(subset=['vote'])
df_without_nan['vote'] = df_without_nan['vote'].str.replace(',', '')
df_without_nan["vote"] = pd.to_numeric(df_without_nan["vote"])
#Divide the votes into buckets
df_without_nan['vote_group'] = pd.cut(df_without_nan['vote'], bins=[-10, 5, 10, 20, 50, 80, 100, 200, 1000, 9000])

#Get the Number of Reviews that are Votes
plt.figure(figsize=(15,8))
sns.countplot(
    x = 'vote_group',
    data = df_without_nan,
    order = df_without_nan.vote_group.value_counts().index
)
plt.xlabel("Number of Votes")
print("Writing to file")
plt.savefig(base_name + '_Vote_Distribution_plot.png')
print("done\n\n")

print("Getting the Word Count Graph")
df['reviewText_word_count'] = df["reviewText"].apply(lambda x: len(str(x).split(' ')))
df['word_count_group'] = pd.cut(df['reviewText_word_count'], bins=[-1, 5, 10, 20, 50, 80, 100, 200, 1000, 9000])
#Get the Distribution of text length
plt.figure(figsize=(15,8))
sns.countplot(
    x = 'word_count_group',
    data = df,
    order = df.word_count_group.value_counts().index
)
plt.xlabel("Number of Words")
plt.title("Number of Words in Review Text")
print("Writing to file")
plt.savefig(base_name + '_Worc_Count_Distribution_plot.png')
print("done\n\n")

print("Success")

