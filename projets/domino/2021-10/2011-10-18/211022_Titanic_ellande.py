from numpy import equal
import pandas as pd
from module1 import p

titanic = pd.read_excel("C:\Dev\Simplon\\1_BasePython\Marwa\\Titanic.xls")
ti1=titanic.copy()
print(titanic.head)

ti1=ti1.drop(['sibsp','ticket','name','fare','cabin','boat','embarked','parch'],axis=1)
print (ti1.shape)
describe=ti1.describe()
print("ti1 : \n ",describe)
""" print(ti1['pclass'].value_counts())"""

ti2=ti1.copy()
ti2=ti2.dropna(subset=['age','sex'])
#print("ti2 : \n ",ti2.groupby(['sex', 'pclass']).mean())
print("/n ti2 : \n ",ti2.describe())
#print(ti1)
print(ti1.groupby(['sex', 'pclass']).mean()) 
print(ti2.groupby(['sex', 'pclass']).mean()) 

body1 = ti1['body']
body1 = body1.dropna()

body2 = ti1['body'].notna()
body3 = ti1[ti1['body'].notna()]  # recupere tout t1 sans les nan de body equivalent du dropna(subset="column")
body4 = ti1['body'].dropna()


p(body1.head)
p(body2.head)
p(body3.head)
p(body4.head)

cat1=ti2.copy()[ti1["age"]<20]
cat2=ti2.copy()
cat2 = cat2[cat2["age"]>20 ]
cat2 = cat2[cat2["age"]<30]
cat2 = cat2.sort_values('age')
cat3=ti2.copy()[(ti2["age"]>30) & (ti2["age"]<40)]
cat4=ti2.copy()[ti1["age"]>40].sort_values('age')

p(cat1)
p(cat2)
p(cat3)
p(cat4)

catage=ti2.copy()
catage.loc[catage['age']<=20,['age']]=0
catage.loc[(catage['age']>20)&(catage['age']<=30),['age']]=1
catage.loc[(catage['age']>30)&(catage['age']<=40),['age']]=2
catage.loc[(catage['age']>40),['age']]=3

print (catage['age'].value_counts())

data=ti2.copy()

data=data[data['sex'].astype('category').cat.codes]
print(data)