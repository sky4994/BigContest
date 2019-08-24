# https://github.com/Swalloow/Kaggle/blob/master/Titanic%20Survivors/Titanic%20RandomForest.ipynb/
# 설명 : https://swalloow.github.io/decison-randomforest
import numpy as np
import pandas as pd


## Load Data


label_raw = pd.read_csv('C:/Users/CPB06GameN/Desktop/PROGRAM/big/2019빅콘테스트_챔피언스리그_데이터_수정/train_label.csv')
activity_raw = pd.read_csv('C:/Users/CPB06GameN/Desktop/PROGRAM/big/2019빅콘테스트_챔피언스리그_데이터_수정/train_activity.csv')


activity_raw.dtypes

activity = activity_raw
label = label_raw
activity_raw.shape

act_grp = activity.groupby('acc_id',as_index=False).sum()
act_label_grp = pd.merge(act_grp, label, on ='acc_id', how='inner')
act_grp.shape, act_label_grp.shape

# act_label = label.merge(activity, on ='acc_id', how='inner')

# act_label_grp = act_label.groupby('acc_id',as_index=False).mean()

print(act_label_grp.shape)
act_label_grp.head()

data = act_label_grp.drop(['day','char_id','amount_spent'], axis =1)
data['survived'] = np.where(data['survival_time'] == 64, 1,0)
# 'survived','death'

data.drop(['acc_id','survival_time'], axis=1, inplace = True)

data.head()
# data.survived.sum()

#훈련세트, 테스트세트 나누기
from sklearn.model_selection import train_test_split

y = data['survived']
x = data.iloc[:,:-1]





train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.3)

train_x.shape, test_x.shape, train_y.shape, test_y.shape


## DecisionTree

from sklearn.tree import DecisionTreeClassifier

decision = DecisionTreeClassifier(max_depth = 7).fit(train_x,train_y)
# print(decision.score(train_x, train_y))
print(decision.score(test_x, test_y))

from sklearn.tree import export_graphviz
import graphviz
import pydot

export_graphviz(decision,
                feature_names = x.columns,
                class_names = ['Death', 'Survived'],
                out_file = 'decisionTree1.dot',
                impurity = False,
                filled = True)

# Encoding 중요
# (graph,) = pydot.graph_from_dot_file('decisionTree1.dot', encoding='utf8')

# Dot 파일을 Png 이미지로 저장
# graph.write_png('decisionTree1.png')
#
with open('decisionTree1.dot') as f:
    dot_graph = str(open("decisionTree1.dot", "rb").read(), 'utf8')

src = graphviz.Source(dot_graph)
src.render('act-lable.gv', view=True)


