param N > 0; # number of rows in data set (44119)
param M > 0; # dimention of location vectors
param K > 0; # number of clusters (7)

param s {1..K, 1..M}; # centroids
param point {1..N, 1..M}; # location
# upper and lower bounds for cluster size
param up > 0;
param low > 0;


var y {1..K, 1..N} >= 0; # is a point assigned to a cluster

minimize Cluster:
	sum {i in 1..K, j in 1..N} 
		y[i,j] * sum {d in 1..M} (s[i,d]*s[i,d] - 
			2*point[j,d]*s[i,d]);

# Each point should belong to exactly one cluster		
subject to Assign_ea_pt {j in 1..N}:
	sum {i in 1..K} y[i,j] = 1;
	
subject to Cluster_size_low {i in 1..K}:
	sum {j in 1..N} y[i,j] >= low;

subject to Cluster_size_hi {i in 1..K}:
	sum {j in 1..N} y[i,j] <= up;