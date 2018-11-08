import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# pyproj is used to regrid the data on a Basemap. 
try:
    from pyproj import Proj       
except:
    # Note to self: Since pyproj is not installed locally, load it this way
    import sys
    sys.path.append('/uufs/chpc.utah.edu/sys/pkg/python/2.7.3_rhel6/lib/python2.7/site-packages/')
    from pyproj import Proj

C = pd.read_csv('data/traffic_accidents.csv', low_memory=False)
date = C['REPORTED_DATE']

# eliminate rows with no accurate geolocation data
xcriteria = C['GEO_LON'] <= -100
ycriteria = C['GEO_LAT'] >= 30

X = C[xcriteria]
Y = C[ycriteria]

m = Basemap(projection='geos', lon_0=-104.991531,
            resolution='i', area_thresh=5000,
            llcrnrx=X.min(),llcrnry=Y.min(),
            urcrnrx=X.max(),urcrnry=Y.max())

plt.figure(figsize=[15, 15])
m.drawcoastlines(color='w')
m.drawcountries(color='w')
m.drawstates(color='w')

plt.title('Denver Traffic Accidents', fontweight='semibold', fontsize=15)
plt.title('%s' % date.min()-date.max(), loc='right')
plt.show()