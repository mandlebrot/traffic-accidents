'''
Script to collect and assign traffic accident data from years (2017-2018). 
Original data taken from denvergov.org/opendata.
Does not take into account severity of incident.
Developed on Windows 10 using Python 2.7.x
by Lauren Hearn, October/November 2018

For more info, see wiki page at 
http://math.ucdenver.edu/~sborgwardt/wiki/index.php/Mapping_Accident_Prone_Intersections
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from mpl_toolkits.basemap import Basemap
import os
from math import ceil, floor

#load data
C = pd.read_csv('data/traffic_accidents.csv', low_memory=False) # shape = (156467, 20)

# eliminate rows with no accurate geolocation data
xcriteria = C['GEO_LON'] <= -100
ycriteria = C['GEO_LAT'] >= 30
date = '2018-01-01 00:00:00'
yearlim = C['REPORTED_DATE'] >= date
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
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Unclustered Accident Data from ' + date)
plt.show()

# set up data
clusters = input('How many clusters would you like to use (try 7)? ')
centroids = df.sample(clusters)
centroids.reset_index(drop=True, inplace=True)
rows, columns = df.shape
a = float(rows)/float(clusters)
upb = int(ceil(a))
lowb = int(floor(a))

print rows, upb, lowb
print centroids

# Use Pyomo
from pyomo.environ import *
from pyomo.dae import *
from pyomo.opt import SolverFactory

m = ConcreteModel()

# Sets, Parameters, and Variables
m.N = RangeSet(0,rows-1)
m.M = RangeSet(0,columns-1)
m.K = RangeSet(0,clusters-1)

# call data from Pandas dataFrame directly
s = centroids.iloc()
point = df.iloc()

m.y = Var(m.K,m.N,domain=Binary)

# Constraints
# Each point should belong to exactly one cluster
def Assign_ea_point(m, j):
    return sum((m.y[i,j]) for i in m.K) == 1
m.Assign = Constraint(m.N, rule=Assign_ea_point)

# Clusters should have upper and lower bounds on number of members
def Cluster_size(m, i):
    return (lowb, sum((m.y[i,j]) for j in m.N), upb)
m.Size = Constraint(m.K, rule=Cluster_size)

# Objective function
def ObjRule(m):
    return sum((m.y[i,j] * sum((s[i,d]*s[i,d]-2*point[j,d]*s[i,d]) for d in m.M)) for i in m.K for j in m.N)
    
m.Obj = Objective(rule=ObjRule, sense=minimize)
opt = SolverFactory('glpk')
results = opt.solve(m)

print results

# plot results
Xmin = X.min()
Xmax = X.max()
Ymin = Y.min()
Ymax = Y.max()

cpalette = ['red','maroon','yellow','olive','lime','green','aqua','blue','fuchsia','purple']
#add group labels
for i in range(1,clusters):
    for j in range(1,rows):
        if m.y[i,j] == 1:
            plt.scatter(point[j,0], point[j,1], c=cpalette[i-1], s=7)
    
plt.xlim(Xmin, Xmax)
plt.ylim(Ymin, Ymax)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Clustered Accident Data from ' + date)
plt.show()
