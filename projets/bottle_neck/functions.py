import matplotlib.pyplot as plt
import seaborn as sns


def get_outliers_datas(df, colname):
    """[summary]

    Args:
        df ([type]): [description]
        colname ([type]): [description]

    Returns:
        (float, float, float, float): q_low, q_hi,iqr, q_min, q_max
    """
    # .quantile(0.25) pour Q1
    q_low = df[colname].quantile(0.25)
    #  .quantité(0.75) pour Q3
    q_hi  = df[colname].quantile(0.75)
    # IQR = Q3 - Q1
    iqr = q_hi - q_low
    # Max = Q3 + (1.5 * IQR)
    q_max = q_hi + (1.5 * iqr)
    # Min = Q1 - (1.5 * IQR)
    q_min = q_low - (1.5 * iqr)
    return q_low, q_hi,iqr, q_min, q_max


def draw_scatter(df, colname):
    _, _, _, q_min, q_max = get_outliers_datas(df, colname)
    df_only_ok = df[(df[colname]>=q_min) & (df[colname]<=q_max)]
    df_only_ouliers = df[(df[colname]<q_min) | (df[colname]>q_max)]

    plt.scatter(df_only_ok[colname].index, df_only_ok[colname].values, c='blue')
    plt.scatter(df_only_ouliers[colname].index, df_only_ouliers[colname].values, c='red')
    plt.show()


def draw_boxplot_v(df, colname):
    plt.figure(figsize=(5,10))
    df.boxplot(column=[colname], grid=True)


def draw_boxplot_h(df, colname):
    plt.figure(figsize=(15,5))
    sns.boxplot(data=df[colname],x=df[colname])

