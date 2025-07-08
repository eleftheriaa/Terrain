import math
import numpy as np

class Edge:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def __eq__(self, other):
        return ({self.p1, self.p2} == {other.p1, other.p2})

    def __hash__(self):
        return hash(frozenset([self.p1, self.p2]))

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return isinstance(other, Point) and self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def __repr__(self):
        return f"Point({self.x}, {self.y})"

def circumcircle_contains(tri, point):
    A, B, C = tri
    ax, ay = A.x, A.y
    bx, by = B.x, B.y
    cx, cy = C.x, C.y
    dx, dy = point.x, point.y

    mat = np.array([
        [ax - dx, ay - dy, (ax - dx)**2 + (ay - dy)**2],
        [bx - dx, by - dy, (bx - dx)**2 + (by - dy)**2],
        [cx - dx, cy - dy, (cx - dx)**2 + (cy - dy)**2],
    ])

    return np.linalg.det(mat) < 0



def super_triangle(points):
    """Creates a super triangle that contains all points."""

    min_x = min(p.x for p in points)
    min_y = min(p.y for p in points)
    max_x = max(p.x for p in points)
    max_y = max(p.y for p in points)

    dx = max_x - min_x
    dy = max_y - min_y
    delta_max = max(dx, dy)
    mid_x = (min_x + max_x) / 2
    mid_y = (min_y + max_y) / 2

    p1 = Point(mid_x - 20 * delta_max, mid_y - delta_max)
    p2 = Point(mid_x, mid_y + 20 * delta_max)
    p3 = Point(mid_x + 20 * delta_max, mid_y - delta_max)

    return np.array([p1, p2, p3])




def delaunay_bowyer_watson(points):
    # Create super triangle
    points = [Point(p[0], p[1]) for p in points]

    st = super_triangle(points)
    triangles = [tuple(np.array(p) for p in st)]# List of triangle point-triplets

    triangulation = [tuple(st)]

    for point in points:
        bad_triangles = []
        for tri in triangulation:
            # print(tri)
            if circumcircle_contains(tri, point):
                bad_triangles.append(tri)
        # print(bad_triangles)

        # Find the boundary of the polygonal hole
        edges = set()
        for tri in bad_triangles:
            for i in range(3):
                e = Edge(tri[i], tri[(i + 1) % 3])
                if e in edges:
                    edges.remove(e)  # Shared edge
                else:
                    edges.add(e)

        # Remove bad triangles
        for tri in bad_triangles:
            triangulation.remove(tri)

        # Re-triangulate the hole
        for edge in edges:
            new_tri = (edge.p1, edge.p2, point)
            triangulation.append(new_tri)

    # Remove triangles connected to super triangle points
    final_tris = []
    for tri in triangulation:
        if any(p in st for p in tri):
            continue
        final_tris.append(tri)

    return final_tris
