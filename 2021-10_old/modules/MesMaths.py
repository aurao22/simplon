for cat in data["categ"].unique():
    subset = data[data.categ == cat]
    print("-"*20)
    print(cat)
    print("moy:\n",subset['montant'].mean())
    print("med:\n",subset['montant'].median())
    print("mod:\n",subset['montant'].mode())
    # Valeur empirique
    print("var:\n",subset['montant'].var(ddof=0))
    # écart-type  empirique
    print("ect:\n",subset['montant'].std(ddof=0))
    # le skewness empirique
    print("skw:\n", subset['montant'].skew())
    #  kurtosis empirique
    print("kur:\n", subset['montant'].kurtosis())
    subset["montant"].hist()
    plt.show()
    subset.boxplot(column="montant", vert=False)
    plt.show()

