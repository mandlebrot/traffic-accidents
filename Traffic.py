'''
Script to collect and assign traffic accident data from years (2017-2018). 
Original data taken from denvergov.org/opendata.
Does not take into account severity of incident.
Developed on Windows 10 using Python 2.7 and AMPL
by Lauren Hearn, October/November 2018
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.cluster import KMeans
from mpl_toolkits.basemap import Basemap
from amplpy import AMPL, Parameter
from amplpy import DataFrame as adf
import os
import math

#load data
C = pd.read_csv('data/traffic_accidents.csv', low_memory=False) # shape = (156467, 20)

# eliminate rows with no accurate geolocation data
xcriteria = C['GEO_LON'] <= -100
ycriteria = C['GEO_LAT'] >= 30
yearlim = C['REPORTED_DATE'] >= '2017-01-01 00:00:00'
criteria = xcriteria & ycriteria & yearlim
C = C[criteria] 

#print C.shape # shape = (44119, 20)

X = C['GEO_LON'].values
Y = C['GEO_LAT'].values
coord = C[['GEO_LAT', 'GEO_LON']].values
df = pd.DataFrame({
    'x': X,
    'y': Y
})

# plot data for sample map
plt.scatter(X, Y, c='black', s=7)
plt.show()

'''
# Kmeans clustering
Xmin = X.min()
Xmax = X.max()
Ymin = Y.min()
Ymax = Y.max()

kmeans = KMeans(n_clusters=10)
kmeans.fit(df)
labels = kmeans.predict(df)
centroids = kmeans.cluster_centers_

fig = plt.figure(figsize=(10, 10))
colmap = {1: 'C0', 2: 'C1', 3: 'C2', 4: 'C3', 5: 'C4', 6: 'C5', 7: 'C6', 8: 'C7', 9: 'C8', 10: 'C9'}
colors = map(lambda x: colmap[x+1], labels)

plt.scatter(df['x'], df['y'], color=colors, alpha=0.5, edgecolor='k')
for idx, centroid in enumerate(centroids):
    plt.scatter(*centroid, color=colmap[idx+1])
plt.xlim(Xmin, Xmax)
plt.ylim(Ymin, Ymax)
plt.show()
'''
# set up data
clusters = input('How many clusters would you like to use (try 7)? ')
centroids = df.sample(clusters)
centroids.reset_index(drop=True, inplace=True)
rows, columns = df.shape
upb = math.ceil(rows/clusters)
lowb = math.floor(rows/clusters)

# Use Pyomo
from pyomo.environ import *
from pyomo.dae import *

m = ConcreteModel()

m.N = RangeSet(rows)
m.M = RangeSet(columns)
m.K = RangeSet(clusters)
m.up = Param(initialize = upb)
m.low = Param(initialize = lowb)

m.s = Param(m.K,m.M,centroids)
m.point = Param(m.N,m.M,df)

m.y = Var(m.K,m.N)

# Each point should belong to exactly one cluster
def Assign_ea_point(m):
    return sum((m.y[i,:] == 1) for i in m.K)
m.Assign = Constraint(rule=Assign_ea_point)

# Clisters should have upper and lower bounds on number of members
def Cluster_size(m):
    return (m.low, sum((m.y[:,j]) for j in m.N), m.up)
m.Size = Constraint(rule=Cluster_size)

# Objective function
def ObjRule(m):
    return sum(((m.y[i,j]) for i in m.K for j in m.N) * sum(m.s[i,d]*m.s[i,d]- 2*m.point[j,d]*m.s[i,d]) for d in m.M)
    
m.Obj = Objective(rule=ObjRule, sense=minimize)

opt.solve(m)


'''
### Clustering through AMPL
# Set solver, AMPL object
ampl = AMPL()
ampl.setOption('solver', 'cplex')

# Read the model
direc = os.getcwd()
direc = direc + '\cluster.mod'
ampl.read('cluster.mod')

## Set parameters with ampl.getParameter() - this is not insignificant!! docs/examples set an inefficient precident!
# Scalar Parameters
N = ampl.getParameter('N')
N.set(rows)
M = ampl.getParameter('M')
M.set(columns)
K = ampl.getParameter('K')
K.set(clusters)

up = ampl.getParameter('up')
up.set(upb)
low = ampl.getParameter('low')
low.set(lowb)

# vectorial params
#centroids['s'] = list(zip(centroids.x,centroids.y)) 
#cent = centroids['s']
#cent = cent.tolist()
#print type(cent)
#dfs = adf.setValues(cent)
xcol = centroids['x']
ycol = centroids['y']
cent = xcol.align(ycol)
s = ampl.getParameter('s')
s.setValues(cent)
print s

point = ampl.getParameter('point')
point.setValues(dfn)

# solve
ampl.solve()
# Get objective entity by AMPL name      
Cluster = ampl.getObjective('cluster')              
print "Objective is:", Cluster.value()
'''

'''
Bibliography:
https://mubaris.com/posts/kmeans-clustering/
http://benalexkeen.com/k-means-clustering-in-python/
https://arxiv.org/pdf/1308.4004.pdf


'''