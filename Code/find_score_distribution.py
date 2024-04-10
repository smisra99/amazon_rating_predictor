import sys
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

#get the csv and make it a dataframe
csv_file = sys.argv[1]
print("read ", csv_file)
df = pd.read_csv(csv_file)
print("done\n\n")

#Create a plot of the bar graph
print("Creating Graph")
plt.figure(figsize=(15,8))
sns.countplot(
    x = 'overall',
    data = df,
    order = df.overall.value_counts().index
)
plt.xlabel("Overall")
plt.title("Review Overall Scores")

split_name = csv_file.split('.')
base_name = split_name[0]

#Writing to file
print("Writing to file")
plt.savefig(base_name + '_score_distribution_plot.png')
print("Success")