import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../modules')
import FilePanda as fp
import GraphiqueUtile as myGraph
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def category_age(age):
    if age <= 20:
        return '<20 ans'
    elif 20 < age <= 30:
        return '20-30 ans'
    elif 30 < age <= 40:
        return '30-40ans'
    else:
        return '+40 ans'


def prepareData(df):
    df = df.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis=1)
    df = df.dropna(subset=['Age', 'Sex'])
    # Ajout d'une colonne avec la catégorie pour l'age
    df["Age_cat"] = df['Age'].map(category_age)
    df["Age_cat"] = df["Age_cat"].astype('category')
    print(df[df['Age'] < 18]["Sex"])

    # Ajout de l'identification des enfants
    df["Sex_cat"] = df["Sex"].copy()
    for i in df.index:
        age = df.loc[i,"Age"]
        if(age < 18):
            df.loc[i,"Sex_cat"] = "Child"

    # Transformation du sexe en catégorie
    df["Sex"] = df["Sex"].astype('category')
    df["Sex_cat"] = df["Sex_cat"].astype('category')
    print(df.head)
    return df


def displayGroupByMean(df):
    # 3. Groupby() et value_counts()
    print("GROUP BY")
    print("------------ Sex ----------------------")
    print(df.groupby(['Sex']).mean())
    print("------------ Age ----------------------")
    print(df.groupby(['Age']).mean())
    print("------------ 'Sex', 'Survived' ----------------------")
    print(df.groupby(['Sex', 'Survived']).mean())
    print("------------ 'Sex', 'Age' ----------------------")
    print(df.groupby(['Sex', 'Age']).mean())
    print("------------ 'Age', 'Survived' ----------------------")
    print(df.groupby(['Age', 'Survived']).mean())
    print("------------ 'Sex', 'Age', 'Survived' ----------------------")
    print(df.groupby(['Sex', 'Age', 'Survived']).mean())


def displayValueCount(df):
    # 3. Groupby() et value_counts()
    print("VALUE COUNTS")
    print("------------ Sex ----------------------")
    print(df.value_counts(['Sex']))
    print("------------ Age ----------------------")
    print(df.value_counts(['Age']))
    print("------------ 'Sex', 'Survived' ----------------------")
    print(df.value_counts(['Sex', 'Survived']))
    print("------------ 'Sex', 'Age' ----------------------")
    print(df.value_counts(['Sex', 'Age']))
    print("------------ 'Age', 'Survived' ----------------------")
    print(df.value_counts(['Age', 'Survived']))
    print("------------ 'Sex', 'Age', 'Survived' ----------------------")
    print(df.value_counts(['Sex', 'Age', 'Survived']))


def displayBarGraphSeaborn(df, xName, yName, hueName, title):
    colors = sns.color_palette(myGraph.getAColorsPaletteSeaborn())
    g = sns.barplot(x=xName,y=yName, hue=hueName, data=df, palette=colors)
    g.set_ylabel(yName)
    g.set_xlabel(xName)
    plt.title(title)
    plt.show()


def displayPieGraphSeaborn(df, label, values, title=""):
    # define Seaborn color palette to use
    colors = sns.color_palette(myGraph.getAColorsPaletteSeaborn())
    df_group = df.groupby(label).mean()
    print(df_group)
    labels = df_group.index
    values = df_group[values]
    # create pie chart
    plt.pie(values, labels=labels, colors=colors, autopct='%.0f%%')
    plt.title(title)
    plt.show()


# 1. Charger vos données dans un DataFrame Pandas
df = fp.load("211027_titanic.csv")
print("BEFORE ------------------")
print(df.head())
df = prepareData(df)
print("AFTER ------------------")
print(df.head(10))

# displayGroupByMean(df)
# displayValueCount(df)

myGraph.showCorrelationSeaborn(df)
myGraph.displayGraph(df, "Pclass", "Survived")
myGraph.displayGraph(df, "Sex_cat", "Survived")
myGraph.displayGraph(df, "Age", "Survived")

df['Age_cat'].value_counts().plot.bar(color=myGraph.getAColor())
plt.show()
df['Age'].hist(color=myGraph.getAColor())
plt.show()

# df.plot.scatter(x='Pclass', y='Age_cat', c='DarkBlue')
# plt.show()

displayBarGraphSeaborn(df, 'Pclass', "Survived", "Sex_cat", "Survivants par classe et sexe")
displayBarGraphSeaborn(df, 'Age_cat', "Survived", "Sex_cat", "Survivants par Age et sexe")
displayBarGraphSeaborn(df, 'Pclass', "Survived", "Age_cat", "Classes par Age et sexe")

sns.set_theme(style="ticks")
sns.pairplot(df, hue="Sex")
plt.show()

displayPieGraphSeaborn(df, ['Pclass'], 'Survived', "Répartition des survivants par classe")
displayPieGraphSeaborn(df, ['Sex_cat'], 'Survived', "Répartition des survivants par Sex")
displayPieGraphSeaborn(df, ['Age_cat'], 'Survived', "Répartition des survivants par Age")


# sns.set_context('paper')
# Ajout d'une catégorie enfant
# create plot
# sns.countplot(x = 'class', hue = 'who', data = titanic, palette = 'magma')
# plt.title('Survivors')
# plt.show()

print("END")


