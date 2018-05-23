from sklearn.cluster import KMeans
import numpy as np 
from pandas import read_csv

names = ['buying', 'maint', 'doors', 'persons', 
        'lug_boot', 'safety', 'class']
data = read_csv("car.data", names=names)

scales = {'vhigh': 10, 'high':5, 'med':3, 'low':1,
          'more': '8', 'small': 1, 'big':5, '5more':'8'}

for c in data.columns:
    try:
        data[c] = data[c].apply(lambda x: scales[x])
    except KeyError:
        data[c] = data[c].apply(lambda x: scales[x] 
                                    if (x=='more' or 
                                        x == '5more') 
                                        else x)

data.doors = data.doors.astype('int64')
data.persons = data.doors.astype('int64')

labels = data['class']
data = data.drop('class', axis=1)
