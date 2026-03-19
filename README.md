

### Input:
Map image containing closed curves (contours).

---

### Part A:

**i)** Detection of the image curves and their polygonal representation. 
**ii)** Triangulation of the map in two dimensions and presentation of the result as a **`terrain`** in three dimensions.
&nbsp;&nbsp;&nbsp;&nbsp; Repeat using **`Delaunay`** triangulation.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**a.** The minimum angle in 3D must be > α degrees.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**b.** The area of each triangle must be < δ.

---

### Part Β:

**iii)** Ability for the user to define:
&nbsp;&nbsp;&nbsp;&nbsp;– The average height of each closed curve.
&nbsp;&nbsp;&nbsp;&nbsp;– The method of height variation for regions within the curve.

**iv)** Coloring of the terrains based on altitude. 
**v)** Calculation and visualization of the **`dual graph`** for each case. 
**vi)** Calculation and visualization of the minimum distance between two random points on the map, using the dual graph.
**vii)** Repetition of (vi) with the constraint that the transition must not have a slope > 10%.


---
