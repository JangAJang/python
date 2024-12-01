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

pd.crosstab(data.Pclass, data.Survived, margins=True).style.background_gradient(cmap='summer_r')

f, ax = plt.subplots(1, 2, figsize=(18, 8))
data['Pclass'].value_counts().plot.bar(['#CD7F32', '#FFDF00', '#232323'], ax=ax[0])
ax[0].set_title('number of passengers by Pclass')
ax[0].set_ylabel('count')
sbn.countplot(x='Pclass', hue='Survived', data=data, ax=ax[1])
ax[1].set_title('Pclasses : Survived vs Dead')
plt.show()

pd.crosstab([data['Sex'], data['Survived']], data['Pclass'], margins=True).style.background_gradient(cmap='summer_r')

sbn.catplot(x='Pclass', y='Survived', hue='Sex', data=data, kind='point')
plt.show()

data['Age'].describe()

sbn.kdeplot(data.loc[data['Survived'] == 1, 'Age'], label='Survived')
sbn.kdeplot(data.loc[data['Survived'] == 0, 'Age'], label='Died')
plt.xlabel('Age')
plt.ylabel('count')

f, ax = plt.subplots(1, 2, figsize=(18, 8))

sbn.violinplot(x='Pclass', y='Age', hue='Survived', data=data, split=True, ax=ax[0])
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(range(0, 110, 10))

sbn.violinplot(x='Sex', y='Age', hue='Survived', data=data, split=True, ax=ax[1])
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0, 110, 10))
plt.show()

data['Initial'] = 0
for i in data :
    data['Initial'] = data.Name.str.extract('([A-Za-z]+)\.')

data['Initial'].replace(
    ['Mlle', 'Mme', 'Ms', 'Dr', 'Major', 'Lady', 'Countess', 'Jonkheer', 'Col', 'Rev', 'Capt', 'Sir', 'Don'],
    ['Mrs', 'Mrs', 'Mrs', 'Mr', 'Mr', 'Mrs', 'Mrs', 'Other', 'Other', 'Other', 'Mr', 'Mr', 'Mr'], inplace=True)

data.groupby('Initial')['Age'].mean()

data.loc[(data.Age.isnull() & (data.Initial == 'Mr')), 'Age']=33
data.loc[(data.Age.isnull() & (data.Initial == 'Mrs')), 'Age']=36
data.loc[(data.Age.isnull() & (data.Initial == 'Master')), 'Age']=5
data.loc[(data.Age.isnull() & (data.Initial == 'Miss')), 'Age']=22
data.loc[(data.Age.isnull() & (data.Initial == 'Other')), 'Age']=46

f, ax = plt.subplots(1, 2, figsize=(18, 8))

data[data['Survived'] == 0].Age.plot.hist(ax=ax[0], bins=20, edgecolor='black', color='red')
ax[0].set_title='Survived = 0'
x1 = list(range(0, 85, 5))
ax[0].set_xticks(x1)

data[data['Survived'] == 1].Age.plot.hist(ax=ax[1], bins=20, edgecolor='black', color='green')
ax[0].set_title='Survived = 1'
x2 = list(range(0, 85, 5))
ax[0].set_xticks(x2)

plt.show()