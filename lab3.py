from vvrpywork.constants import Key, Mouse, Color
from vvrpywork.scene import Scene2D
from vvrpywork.shapes import (
    Point2D, Line2D, Triangle2D, Circle2D, Rectangle2D,
    PointSet2D, LineSet2D, Polygon2D
)

from random import random

WIDTH = 800
HEIGHT = 800

class Lab3(Scene2D):
    def __init__(self):
        super().__init__(WIDTH, HEIGHT, "Lab3")
        self.points:list[Point2D] = []
        self.triangles:dict[str, Triangle2D] = {}

        self.reset() # scene initialiazation

    def reset(self):
        self.violations = 0
        self.run_task_3 = False

        self.run_task_4 = False
        self.run_task_5 = False

        A = Point2D((-0.8, -0.8))
        B = Point2D((0.8, -0.8))
        C = Point2D((0, 0.8))

        self.points.append(A)
        self.points.append(B)
        self.points.append(C)

        big_triangle = Triangle2D(A, B, C)
        name = str(random())
        self.names=[]
        self.triangles[name] = big_triangle

        self.addShape(A)
        self.addShape(B)
        self.addShape(C)

        self.addShape(big_triangle, name)

    def on_mouse_press(self, x, y, button, modifiers):
        # Set all previous triangles to black
        for name in self.triangles:
            self.triangles[name].color = Color.BLACK
            self.updateShape(name)

        # Process new point
        self.processPoint(Point2D((x, y)))

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        new_pt = Point2D((x, y))
        if self.points[-1].distanceSq(new_pt) > 0.05:
            self.processPoint(new_pt)

    def on_key_press(self, symbol, modifiers):
        if symbol == Key._3:
            self.run_task_3 = True
            # self.flag3 =True
            self.processPoint()
        if symbol == Key._4:
            self.run_task_4 = True
            self.processPoint()
        if symbol == Key._5:
            self.run_task_5 = True
            self.processPoint()

    def processPoint(self, point:Point2D|None = None):
        # Remove shapes from last click
        for i in range(self.violations):
            self.removeShape(f"vc{i}")
            self.removeShape(f"vt{i}")
        self.violations = 0
        if not self.run_task_3 and not self.run_task_4 and not self.run_task_5:
            self.names=[]
            # [Check 1]
            # Check whether a point already exists in the same coords.
            for p in self.points:
                if p.x == point.x and p.y == point.y:
                    return
                
            # Find enclosing triangle.
            count_enclosing = 0
            for name in self.triangles:
                if self.triangles[name].contains(point):
                    count_enclosing += 1
                    name_enclosing = name

            # [Check 2]
            # If no enclosing triangle was found.
            # Or if more than one were found
            if count_enclosing != 1:
                return

            self.points.append(point)
            self.addShape(point)

            

            # TASK 0:
            #   - Create the 3 subdivision triangles and store them to `new_triangles`.
            #   - Remove the enclosing triangle and store the new ones with different colour.
            #
            # HINTS:
            #   - To delete a triangle from `self.triangles`:
            #       del self.triangles[key_to_delete]
            #
            # VARIABLES:
            #   - point: Point2D                   # New point
            #   - name_enclosing: str              # key of the enclosing triangle
            #                                        (both in self.triangles and in the scene)
            #   - new_triangles: list[Triangle2D]  # The 3 triangles that will replace the enclosing.

            new_triangles:list[Triangle2D] = []

            # example
            # new_triangles.append(Triangle2D(point, (point.x + 0.1, point.y), (point.x, point.y + 0.1), color=Color.RED))
            # name = str(random())
            # self.triangles[name] = new_triangles[0]
            # self.addShape(new_triangles[0], name)

            #1. parse the 3 vertices of the eclosing triangle
            p1 = self.triangles[name_enclosing].getPoint1()
            p2 = self.triangles[name_enclosing].getPoint2()
            p3 = self.triangles[name_enclosing].getPoint3()
            print(p1,p2,p3)

            #2. create the new triangles and store them in the "new_triangle"

            new_triangles.append(Triangle2D(point, p1, p2, color = Color.RED))
            new_triangles.append(Triangle2D(point, p3, p2, color = Color.RED))
            new_triangles.append(Triangle2D(point, p3, p1, color = Color.RED))

            #3. remove the enclosing triangle

            self.removeShape(name_enclosing)
            del self.triangles[name_enclosing]

            #4. add the new triangles in dict and scene
            
            for t in new_triangles:
                name = str(random())
                self.triangles[name] = t
                self.names.append(name)
                self.addShape(t,name)

            # TASK 2:
            #   - Check the 3 new triangles for Delaunay violations.
            #   - If not Delaunay, add it with different color and show CircumCircle.
            #
            # HINTS:
            #   - use isDelaunay()
            print("NEW",self.triangles[self.names[-1]])

            for new_triangle in new_triangles:
                if findViolations(self.triangles, new_triangle):
                    showViolations(new_triangle, self, (1, 1, 1, 0.5))

        if self.run_task_3:
            new_triangles3 = [self.triangles[name] for name in self.names[-3:]]

            for new_triangle in new_triangles3:
                if findViolations(self.triangles, new_triangle):
                    
                    p1 = new_triangle.getPoint1()
                    p2 = new_triangle.getPoint2()
                    p3 = new_triangle.getPoint3()

                    if (self.points[-1] == p1) : ep,eq = p2,p3
                    elif (self.points[-1] == p2) :ep,eq = p1,p3
                    else:ep,eq = p2,p2

                    tri_adj_key3, opp_ver3 = findAdjacentTriangle(self.triangles,ep,eq)

                    if tri_adj_key3:
                        current_key = None
                        for key, triangle in self.triangles.items():
                            if triangle == new_triangle:
                                current_key = key
                                break
                        
                        self.removeShape(current_key)
                        self.removeShape(tri_adj_key3)
                        del self.triangles[current_key]
                        del self.triangles[tri_adj_key3]

                                
                        # flip
                        new_tri1 = Triangle2D(self.points[-1], ep, opp_ver3)
                        new_tri2 = Triangle2D(self.points[-1], eq, opp_ver3)
                        
                        name1 = str(random())
                        name2 = str(random())
                        self.triangles[name1] = new_tri1
                        self.triangles[name2] = new_tri2
                        self.addShape(new_tri1, name1)
                        self.addShape(new_tri2, name2)
                        
                        self.names.extend([name1, name2])



            self.run_task_3 = False

        if self.run_task_4:
            stack = [self.triangles[name] for name in self.names[-3:]] if self.names else []
            # new_triangles3 = [self.triangles[name] for name in self.names[-3:]]

            while stack :
                new_triangle = stack.pop()
                 
                if findViolations(self.triangles, new_triangle):
                    
                    p1 = new_triangle.getPoint1()
                    p2 = new_triangle.getPoint2()
                    p3 = new_triangle.getPoint3()

                    if (self.points[-1] == p1) : ep,eq = p2,p3
                    elif (self.points[-1] == p2) :ep,eq = p1,p3
                    else:ep,eq = p2,p2

                    tri_adj_key3, opp_ver3 = findAdjacentTriangle(self.triangles,ep,eq)

                    if tri_adj_key3:
                        current_key = None
                        for key, triangle in self.triangles.items():
                            if triangle == new_triangle:
                                current_key = key
                                break
                        
                        self.removeShape(current_key)
                        self.removeShape(tri_adj_key3)
                        del self.triangles[current_key]
                        del self.triangles[tri_adj_key3]

                                
                        # flip
                        new_tri1 = Triangle2D(self.points[-1], ep, opp_ver3)
                        new_tri2 = Triangle2D(self.points[-1], eq, opp_ver3)
                        
                        name1 = str(random())
                        name2 = str(random())
                        self.triangles[name1] = new_tri1
                        self.triangles[name2] = new_tri2
                        self.addShape(new_tri1, name1)
                        self.addShape(new_tri2, name2)
                        
                        self.names.append(name1)
                        self.names.append(name2)
                        print(self.names)
                        showViolations(new_tri1,self, (0, 1, 0, 0.2) )
                        showViolations(new_tri2,self, (0, 0, 1, 0.2) )
                        
                        stack.append(new_tri1)
                        stack.append(new_tri2)


            self.run_task_4 = False

        if self.run_task_5:
            centroids = {}  # Store centroids by triangle key
            lines = []      # Store lines between centroids
            
            # Step 1: Calculate centroids for all triangles
            for key, triangle in self.triangles.items():
                A = triangle.getPoint1()
                B = triangle.getPoint2()
                C = triangle.getPoint3()
                
                centroid_x = (A.x + B.x + C.x) / 3
                centroid_y = (A.y + B.y + C.y) / 3
                centroid = Point2D((centroid_x, centroid_y),size =1.5,color=(0.5, 0, 0))
                
                centroids[key] = centroid
                self.addShape(centroid)  # Visualize centroid points

            count = 0
            for key, triangle in self.triangles.items():
                A = triangle.getPoint1()
                B = triangle.getPoint2()
                C = triangle.getPoint3()
                
                for edge in [(A, B), (B, C), (C, A)]:
                    adj_key, adj_xy = findAdjacentTriangle(self.triangles, edge[0], edge[1])
                    
                    if adj_key and (adj_key in centroids):
                        if (adj_key, key) not in lines: 
                            line = Line2D(centroids[key], centroids[adj_key],color=(0, 0.5, 0))
                            lines.append((key, adj_key)) 
                            self.addShape(line) 

            
            self.run_task_5 = False

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
    c.radius *= 0.99  # Shrink the circle a bit in order to exclude points of its circumference.
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

def showViolations(tri:Triangle2D, scene:Lab3, col:Color):
    c = tri.getCircumCircle()
    c.color = col
    scene.addShape(c, f"vc{scene.violations}")

    filled = Triangle2D(tri.getPoint1(), tri.getPoint2(), tri.getPoint3(), color=col, filled=True)
    scene.addShape(filled, f"vt{scene.violations}")
    scene.violations += 1

if __name__ == "__main__":
    app = Lab3()
    app.mainLoop()
