from vvrpywork.constants import Key, Mouse, Color
from vvrpywork.scene import Scene2D
from vvrpywork.shapes import (
    Point2D, Line2D, Triangle2D, Circle2D, Rectangle2D,
    PointSet2D, LineSet2D, Polygon2D, PointSet3D, Point3D
)

from random import random
import numpy as np

def findAdjacentTriangle3D(pointset: PointSet3D, names:list , p1:Point3D, p2:Point3D, exclude_key: str = None) -> tuple[str,Point2D]:
    tri_adj_key = ""
    opp_ver = None

    for i in range(len(pointset)):    
        for j,p in enumerate(pointset[i]):
    
            v1 = p[0]
            v2 = p[1]
            v3 = p[2]

            name= names[i][j]
            if name == exclude_key:
                continue

    
            if (p1 == v1 and p2 == v2 ) or (p1 == v2 and p2 == v1 ) : 
                tri_adj_key=name
                opp_ver=v3
                break
            if (p1 == v1 and p2 == v3 ) or (p1 == v3 and p2 == v1 ) : 
                tri_adj_key=name
                opp_ver=v2
                break
            if (p1 == v3 and p2 == v2 ) or (p1 == v2 and p2 == v3 ) : 
                tri_adj_key=name
                opp_ver=v1
                break
            
        
    return tri_adj_key, opp_ver

def findAdjacentTriangle(tris:dict[str,Triangle2D], p1:Point2D, p2:Point2D) -> tuple[str,Point2D]:
    tri_adj_key = ""
    opp_ver = None

    # TASK 1:
    #   - Find a triangle that contains p1-p2.
    #   - Save its key in `tri_adj_key` and the remaining vertex in `opp_ver`.

    for key in tris:
        v1 = tris[key].getPoint1()
        v2 = tris[key].getPoint2()
        v3 = tris[key].getPoint3()

        if (p1 == v1 and p2 == v2 ) or (p1 == v2 and p2 == v1 ) : 
            tri_adj_key=key
            opp_ver=v3
            break
        if (p1 == v1 and p2 == v3 ) or (p1 == v3 and p2 == v1 ) : 
            tri_adj_key=key
            opp_ver=v2
            break
        if (p1 == v3 and p2 == v2 ) or (p1 == v2 and p2 == v3 ) : 
            tri_adj_key=key
            opp_ver=v1
            break
        
        
    return tri_adj_key, opp_ver

def isDelaunay(t:Triangle2D, p:Point2D) -> bool:
    '''Checks if `t` is a Delaunay triangle w.r.t `p`.'''

    c = t.getCircumCircle()
    c.radius *= 0.97  # Shrink the circle a bit in order to exclude points of its circumference.
    if c.contains(p):
        return False
    return True

def findViolations(all_triangles:dict[str,Triangle2D], new_triangle:Triangle2D) -> bool:
    '''Checks if the given triangle is Delaunay.

    Checks if a triangle is delaunay, checking all its adjacent
    triangles.

    Args:
        all_triangles: A dictionary of all the triangles.
        new_triangle: The triangle to check.
    
    Returns:
        False if the given triangle is delaunay and True otherwise.
    '''

    is_delaunay = True
    # 1. use findAdjacentTriangle() to check whether new_triangle is Delaunay
    tri_adj_key, opp_ver = findAdjacentTriangle(all_triangles,new_triangle.getPoint1(),new_triangle.getPoint2())
    if tri_adj_key:
        if not isDelaunay(new_triangle, opp_ver): 
            is_delaunay = False
    tri_adj_key, opp_ver = findAdjacentTriangle(all_triangles,new_triangle.getPoint3(),new_triangle.getPoint2())
    if tri_adj_key:
        if not isDelaunay(new_triangle, opp_ver):
            is_delaunay = False

    tri_adj_key, opp_ver = findAdjacentTriangle(all_triangles,new_triangle.getPoint1(),new_triangle.getPoint3())
    if tri_adj_key:
        if not isDelaunay(new_triangle, opp_ver):
            is_delaunay = False

    # 2. use function isDelaunay()



    return not is_delaunay


def if_not_delaunny(triangle, all_triangles) -> list[str, Triangle2D, str, Triangle2D]:
    # Get triangle vertices
    p1, p2, p3 = triangle.getPoint1(), triangle.getPoint2(), triangle.getPoint3()
    edges = [(p1, p2), (p2, p3), (p3, p1)]
    third_vertices = [p3, p1, p2]  # Opposite vertex to each edge

    for i, edge in enumerate(edges):
        third_v = third_vertices[i]
        adj_key, opp_vertex = findAdjacentTriangle(all_triangles, edge[0],edge[1])
        if adj_key is None or opp_vertex is None:
            continue  # Skip if no adjacent triangle found

        # Flip edge: create two new triangles with the opposite point
        new_tri1 = Triangle2D(opp_vertex, edge[0], third_v, color=Color.RED)
        new_tri2 = Triangle2D(opp_vertex, edge[1], third_v, color=Color.RED)

        name1 = str(random())
        name2 = str(random())

    return name1, new_tri1, name2, new_tri2  # Return first valid flip


def find_closest_vertex( vertices, query: tuple) -> int:

    difference = (vertices - query)
    dist = (difference * difference).sum(axis=1)

    closest_vertex_index = np.argmin(dist)

    return closest_vertex_index
