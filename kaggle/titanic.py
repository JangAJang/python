import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
import warnings

plt.style.use('seaborn-v0_8-darkgrid')
warnings.filterwarnings('ignore')

data = pd.read_csv('/kaggle/input/titanic/train.csv')

data.isnull().sum() # count is null tuples

f, ax = plt.subplots(1, 2, figsize = (18, 8)) #row, col, total figsize
data['Survived'].value_counts() #does not count null-values
data['Survived'].value_counts().plot.pie(explode = [0, 0.1], autopct = '%1.1f%%', ax = ax[0], shadow=True)
ax[0].set_title('Survived')
ax[0].set_ylabel('')
sbn.countplot(x = 'Survived', data = data, ax = ax[1])
ax[1].set_title('Survived')
plt.show()

data.groupby(['Sex', 'Survived'])['Survived'].count()

f, ax = plt.subplots(1, 2, figsize = (18, 8)) #row, col, total figsize
data[['Sex', 'Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs Sex')
sbn.countplot(x = 'Sex', hue='Survived', data=data, ax=ax[1])
ax[1].set_title('Sex: Survived vs Dead')
plt.show()