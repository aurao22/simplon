import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def displayInfo(df):
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("shape:", df.shape)
    print("line:", df.index)
    print("col :", df.columns)
    print("Describe    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(df.describe)
    print("Describe () ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(df.describe(include="all"))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


# 1. Charger vos données dans un DataFrame Pandas
def load(file):
    # Charger vos données dans un DataFrame Pandas
    df = pd.read_csv(file)
    displayInfo(df)
    print("head")
    print(df.head())
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    return df


# 2. Nettoyer votre Dataset avec drop(), dropna() et fillna()
def clean(df):
    df_drop = df.copy(deep=False)
    # Supprimer les colonnes inutiles
    #df_drop = df_drop.drop(['sibsp', 'ticket', 'name', 'fare', 'cabin', 'boat', 'embarked', 'parch'], axis=1)
    print("DROP ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    df_drop = df_drop.drop(['PClass'], axis=1)
    displayInfo(df_drop)

    print("DROP NA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    df_drop = df_drop.dropna(subset=['Age', 'Sex'])
    displayInfo(df_drop)

    print("DUPLICATED ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    df_drop = df_drop.drop_duplicates(subset=['Name'])
    displayInfo(df_drop)

    print("FILL NA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # Ne sert à rien après un drop NA puisque les données ayant un NaN ont été supprimées
    df_drop = df_drop.fillna(-1)
    displayInfo(df_drop)

    print("END CLEAN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    return df_drop



# 1. Charger vos données dans un DataFrame Pandas
df = load("Titanic.csv")
# 2. Nettoyer votre Dataset avec drop(), dropna() et fillna()
df_drop = clean(df)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

# 3. Groupby() et value_counts()
print("VALUE COUNTS")
print("------------ Sex ----------------------")
print(df_drop.value_counts(['Sex']))
print("------------ Age ----------------------")
print(df_drop.value_counts(['Age']))
print("------------ 'Sex', 'Survived' ----------------------")
print(df_drop.value_counts(['Sex', 'Survived']))
print("------------ 'Sex', 'Age' ----------------------")
print(df_drop.value_counts(['Sex', 'Age']))
print("------------ 'Age', 'Survived' ----------------------")
print(df_drop.value_counts(['Age', 'Survived']))
print("------------ 'Sex', 'Age', 'Survived' ----------------------")
print(df_drop.value_counts(['Sex', 'Age', 'Survived']))

print("GROUP BY")
print("------------ Sex ----------------------")
print(df_drop.groupby(['Sex']).mean())
print("------------ Age ----------------------")
print(df_drop.groupby(['Age']).mean())
print("------------ 'Sex', 'Survived' ----------------------")
print(df_drop.groupby(['Sex', 'Survived']).mean())
print("------------ 'Sex', 'Age' ----------------------")
print(df_drop.groupby(['Sex', 'Age']).mean())
print("------------ 'Age', 'Survived' ----------------------")
print(df_drop.groupby(['Age', 'Survived']).mean())
print("------------ 'Sex', 'Age', 'Survived' ----------------------")
print(df_drop.groupby(['Sex', 'Age', 'Survived']).mean())







