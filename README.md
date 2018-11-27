# traffic-accidents
Script to collect and assign traffic accident data to balanced clusters.<br> 
Original data taken from the [Denver Open Data Catalog](https://www.denvergov.org/opendata/).<br>
Does not take into account severity of incident.
<br><br>
Files Include:
* Original AMPL model (.mod) file
* Python script to cluster and plot data
* Poster presented at the 2018 Data to Policy Symposium at the University of Colorado, Denver
<br><br>
To Do:
* Narrow data to smaller areas to observe specific intersection accident density
* Get code to iterate and improve centroid locations automatically
* Run with CPLEX solver (more efficient for Mixed-Integer LPs)
<br><br>
Developed on Windows 10 using Python 2.7.x
by Lauren Hearn, October/November 2018<br>
For more info, see [wiki page](http://math.ucdenver.edu/~sborgwardt/wiki/index.php/Mapping_Accident_Prone_Regions).
