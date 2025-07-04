import cv2
import numpy as np
from vvrpywork.constants import Key, Mouse, Color
from vvrpywork.scene import Scene2D, Scene3D
from vvrpywork.shapes import Point2D, PointSet2D, LineSet2D,Polygon2D, Line2D, Triangle2D,Point3D, PointSet3D,LineSet3D,Line3D
from contour_map_function import generate_contour_function, generate_contour_image
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from random import random
from delaunny import findViolations, if_not_delaunny
from scipy.spatial import Delaunay
import math
N =15
WIDTH = 950
HEIGHT = 950
COLORS = [Color.RED, Color.MAGENTA,   Color.ORANGE, Color.YELLOW, Color.YELLOWGREEN ,Color.GREEN,Color.DARKGREEN, Color.BLUE, Color.BLACK, Color.CYAN,Color.GRAY , Color.WHITE]
H_STEP = 0.1
S = Scene3D
class Terrain(S):
    
    def __init__(self):
        super().__init__(WIDTH, HEIGHT, "Terrain")
        # self.set_slider_value(0, H_STEP)
        self.triangles:dict[str, Triangle2D] = {}
        self.per_contour_triangles = {}
        self.names = []
        # self.all_triangles = []
        # self.all_triangles_keys = []
        self.violations = 0

        self.pointlist2d = []
        self.new_triangles:dict[str, Triangle2D] = {}
        self.filled_tris:dict[str, Triangle2D] = {} 

    # def on_slider_change(self, slider_id, value):
    #     if slider_id == 0:
    #         self.step = value 

    def on_key_press(self, symbol, modifiers):
        if symbol == Key._1:
            self.task1()

        if symbol == Key._2:
            self.task2()
        
        if symbol == Key.D:
            self.check_delaunay_status()
        if symbol == Key._3:
            self.task2a()
        if symbol == Key.R :
            self.violations = 0

            for k in self.names:
                self.removeShape(k)
            self.names = []
            for key in self.triangles.keys():
                self.removeShape(key)
            self.triangles = {}
            for key,value in self.filled_tris.items():
                value.filled=False
                self.removeShape(key)
            self.filled_tris = {}

        if symbol == Key._4:
            self.task2b()

    def resample_contour(self, contour, num_points):
        """Uniformly resample contour to have exactly `num_points` points."""
        contour = contour[:, 0, :]  # from shape (N, 1, 2) to (N, 2)
        contour_len = cv2.arcLength(contour, True)
        epsilon = 0.001 * contour_len
        approx = cv2.approxPolyDP(contour, epsilon, True)

        contour_float = approx.astype(np.float32)
        total_dist = np.cumsum([0] + [np.linalg.norm(contour_float[i] - contour_float[i - 1]) for i in range(1, len(contour_float))])
        interp_dists = np.linspace(0, total_dist[-1], num_points)
        resampled = []

        for d in interp_dists:
            i = np.searchsorted(total_dist, d)
            if i == 0:
                resampled.append(contour_float[0])
            else:
                p1 = contour_float[i - 1]
                p2 = contour_float[i]
                t = (d - total_dist[i - 1]) / (total_dist[i] - total_dist[i - 1])
                resampled.append((1 - t) * p1 + t * p2)

        resampled = np.array(resampled).squeeze().tolist()
        # print(resampled)
        new_resampled = []
        for x,y in resampled:
            new_resampled.append([round(x), round(y)])
        # print(new_resampled)

        return new_resampled             


    def task1(self):
        filename = generate_contour_image()

        # img = cv2.imread("contour_maps/pain2t.png")
        # img = cv2.imread("contour_maps/paint3.png")

        # img = cv2.imread("contour_maps/blurred2.png")
        img = cv2.imread("contour_maps/bandw.png")

        # img = cv2.imread("contour_maps/to_draw.png")


        # img = cv2.imread(filename)



        # ***************** DETECT CONTOURS **********************
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([50, 50, 50])
        mask = cv2.inRange(img, lower_black, upper_black)
        # Convert to boolean and skeletonize
        skeleton = skeletonize(mask > 0)
        skeleton_uint8 = (skeleton * 255).astype(np.uint8)
        # Find contours
        # cv2.CHAIN_APPROX_SIMPLE: stores essential points (like corners)
        contours, _ = cv2.findContours(skeleton_uint8, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) 
        half_contours = contours[::2]
        count_points = 0
        # *************** DRAW IMAGE *********************
        # Image shape for normalization
        w,h =img.shape[1], img.shape[0]


        

        # sort contours from innermost to outermost
        sorted_contours = half_contours[::-1]
        print(f"sorted contours:{len(sorted_contours)} ")
        print(f"contours:{len(half_contours)} ")

        # *************** DRAW CONTOURS *****************
        pointlist2d = []
        self.pointlist2d =pointlist2d
        for i, c in enumerate(half_contours):
            pointset = PointSet2D(size=0.5,color=Color.RED)
            lineset = LineSet2D(width = 0.5)
            nx_minus1, ny_minus1 = None, None
            # *************** RESAMPLE CONTOURS *****************
            rcontour = self.resample_contour(c, N *(i+1))
            # print(f"contour_{i}", rcontour)
            # print(rcontour)
            for x_pixel, y_pixel in rcontour:
                # Normalize pixel coords to [-1, 1]
                nx = 2 * (x_pixel / w) - 1
                ny = 2 * (y_pixel / h) - 1
                pn = Point2D([nx,ny],color=Color.RED)
                pointset.add(pn)
                
                if nx_minus1 is not None and ny_minus1 is not None: 
                    pn_minus1 = Point2D([nx_minus1,ny_minus1],color=Color.RED)
                    lineset.add(Line2D(pn,pn_minus1,color=Color.BLACK))
                nx_minus1, ny_minus1 = nx, ny
                count_points +=1 
            self.names.append(f"contour_{i}")

            # self.addShape(lineset)
            # self.addShape(pointset, self.names[i])
            pointlist2d.append(pointset)
            # count += 1
        # print("number of points in all contour",pointlist2d)   
        return pointlist2d
    
    def closest_point_in_contour(self, point, A):
        """
        Given a point with .x, .y and a contour A (list of points with .x, .y),
        returns indices of the closest and second closest points in A to the given point.
        """
        dists = []
        for i, a in enumerate(A):
            dist = np.linalg.norm(np.array([a.x, a.y]) - np.array([point.x, point.y]))
            dists.append((i, dist))
        
        # Sort by distance
        dists.sort(key=lambda x: x[1])
        
        # Return indices of closest and second closest points
        closest_idx = dists[0][0]
        second_closest_idx = dists[1][0]
        return closest_idx, second_closest_idx
    
    def angle_between(self, x1, x2, x3):
        # Returns angle at p2 between vectors p2->p1 and p2->p3
        def vec(a, b):
            return (b[0] - a[0], b[1] - a[1])
        def dot(u, v):
            return u[0]*v[0] + u[1]*v[1]
        def norm(v):
            return math.sqrt(v[0]**2 + v[1]**2)
        p1=(x1.x,x1.y)
        p2=(x2.x,x2.y)
        p3=(x3.x,x3.y)


        u = vec(p2, p1)
        v = vec(p2, p3)
        cos_theta = dot(u, v) / (norm(u) * norm(v) + 1e-8)
        angle = math.acos(max(min(cos_theta, 1), -1))
        return math.degrees(angle)
    
    def min_angle_of_triangle(self, x1, x2, x3):
        angle1 = self.angle_between(x2, x1, x3)
        angle2 = self.angle_between(x1, x2, x3)
        angle3 = self.angle_between(x1, x3, x2)
        return min(angle1, angle2, angle3)
    
    def vector(self, p1, p2):
        return (p2[0] - p1[0], p2[1] - p1[1])
    
    def ccw(self,a, b, c):
        # True if points a, b, c are arranged counter-clockwise
        return (c.y - a.y) * (b.x - a.x) > (b.y - a.y) * (c.x - a.x)

    def segments_intersect(self, edge1, edge2):
        p1,p2 = edge1
        p3,p4 = edge2

        return (self.ccw(p1, p3, p4) != self.ccw(p2, p3, p4)) and (self.ccw(p1, p2, p3) != self.ccw(p1, p2, p4))
    
    def triangulate_between(self, A, B):
        inner_tr_dict = {}
        edges =[]
        orphan_a_points  = []
        orphan_b_points  = []

        K = len(A.points)
        L = len(B.points)
        for i in range(L):
            b_curr = B[i]
            b_next = B[(i+1)%L]
            a_idx , a_second_idx = self.closest_point_in_contour(b_curr, A)
            a_curr= A[a_idx]
            a_curr_second= A[a_second_idx]

            a_next = A[(a_idx +1)%K]
           
            # line = Line2D(b_curr, a_curr ,0.3,Color.MAGENTA)
            # self.addShape(line)

            alpha = self.angle_between(a_curr, b_curr, b_next)
            beta = self.angle_between(a_curr, a_next, b_next)
            name1 = str(random()) 
            name2 = str(random()) 
            tri1 = None
            tri2 = None

            if alpha < beta:
                cur_edges = [(a_curr, b_next), (a_curr, b_curr), (a_next, b_next)]
                if all(not self.segments_intersect(cur_edge, edge) for cur_edge in cur_edges for edge in edges):
                    tri1 = Triangle2D( b_curr, b_next,a_curr ,0.3, color=Color.DARKGREEN)
                    tri2 = Triangle2D(a_curr, a_next, b_next, 0.3, color=Color.DARKGREEN)
                    edges.extend([(a_curr, b_next), (a_curr, b_curr), (a_next, b_next)])
                    
                
            else:
                cur_edges = [(b_curr, a_next), (a_curr, b_curr), (a_next, b_next)]
                if all(not self.segments_intersect(cur_edge, edge) for cur_edge in cur_edges for edge in edges):
                    tri2 = Triangle2D(a_curr,a_next ,b_curr , 0.3, color=Color.MAGENTA)
                    tri1 = Triangle2D(b_curr, b_next,  a_next, 0.3, color=Color.MAGENTA)
                    edges.extend([(b_curr, a_next), (a_curr, b_curr), (a_next, b_next)])
                

            if tri1 and tri2:
                # self.addShape(tri1, name1)
                # self.addShape(tri2, name2)
                self.triangles[name1] = tri1
                self.triangles[name2] = tri2
                inner_tr_dict[name1] = tri1
                inner_tr_dict[name2] = tri2
            else:
                orphan_a_points.append(a_curr)
                orphan_b_points.append(b_curr)


                # print(f"Skipped point {i} due to intersecting edges")
        
        contour_id = f"contour{len(self.per_contour_triangles) + 1}"
        self.per_contour_triangles[contour_id] = inner_tr_dict

        # for i in range(L):
            
        #     b_curr = B[i]
        #     if (b_curr in orphan_b_points):
        #         orphan_name = str(random())
        #         b_next = B[(i+1)%L]
        #         a_idx_close,a_s = self.closest_point_in_contour(b_curr, orphan_a_points)
        #         print(a_idx_close)
        #         a_close = orphan_a_points[a_idx_close]
        #         tri_orphan = Triangle2D(b_curr, b_next,  a_close, 0.3, color=Color.CYAN)
        #         self.addShape(tri_orphan,orphan_name)

        #         self.triangles[orphan_name] = tri_orphan

        

        
            # tri1 =((b_curr.x,b_curr.y), (a_curr.x,a_curr.y),(a_prev.x,a_prev.y) )
            # tri2 =((b_curr.x,b_curr.y), (a_curr.x,a_curr.y),(a_next.x,a_next.y) )

            # tri1_2D = Triangle2D(b_curr, a_curr, a_prev, 0.3, Color.DARKGREEN)
            # tri2_2D = Triangle2D(b_curr, a_curr, a_next, 0.3, Color.DARKGREEN)
            
            # str1 = f"tr1_{i}"
            # str2 = f"tr2_{i}"
            # angle1 = self.triangle_min_angle(*tri1)
            # angle2 = self.triangle_min_angle(*tri2)
            # print("angle1:",angle1)
            # print("angle2:",angle2)

            # if angle1 > 0:
            #     self.addShape(tri1_2D,str1)
            # elif  angle2 > 0:
            #     self.addShape(tri2_2D,str2)
            # else:
            #     # Triangle too skinny, skip it
            #     continue
            # self.addShape(line2)
            # self.addShape(line3)


        #     print(i)
        #     b_curr = B[i]
        #     b_next = B[(i + 1) % L]
        #     b_next_next = B[(i + 2) % L]
        #     if (i%L==0):
        #         a_idx = self.closest_point_in_contour(b_curr,A)
        #         a_start = a_idx
        #     a_curr = A[(a_idx) % K]
        #     a_next = A[(a_idx + 1) % K]
        #     a_next_next = A[(a_idx + 2) % K]

        #     # Form 3 triangles: (a_curr, a_next, b_curr) and (b_curr, b_next, a_next)
        #     new_tri1 = Triangle2D(b_curr, b_next, a_curr,width=0.5,color = Color.BLUE)
        #     new_tri2 = Triangle2D(a_curr, a_next, b_next,width=0.5,color = Color.BLUE)

        #     new_tri3 = Triangle2D(b_next_next, b_next, a_next,width=0.5,color = Color.BLUE)
        #     new_tri4 = Triangle2D( a_next, a_next_next,b_next_next, width=0.5,color = Color.GREEN)

        #     for tri in [new_tri1, new_tri2, new_tri3, new_tri4]:
        #         name = str(random()) 
        #         inner_tr_dict[name] = tri
        #         self.triangles[name] = tri
        #         self.addShape(tri, name)

            
        #     a_idx +=2
        # contour_id = f"contour{len(self.per_contour_triangles) + 1}"
        # self.per_contour_triangles[contour_id] = inner_tr_dict
        # self.all_triangles.append(inner_tr)
        # self.all_triangles_keys.append(inner_tr_key)

    def check_delaunay_status(self) :
        if (self.triangles == {}):
            triangles = self.new_triangles
        else: triangles = self.triangles
        # Optionally reset if you're using self.names to store last-inserted triangle names
        triangle_keys = list(triangles.keys())
        counter=0
        for i, key in enumerate(triangle_keys):
            triangle = triangles[key]
            is_delaunay = not findViolations(triangles, triangle)

            if (is_delaunay == True ): counter +=1
            else:
                self.showViolations(triangle)
                # circle = triangle.getCircumCircle()
                filled = Triangle2D(triangle.getPoint1(), triangle.getPoint2(), triangle.getPoint3(), color=Color.ORANGE, filled=True)
                name_filled = str(random()) 
                self.filled_tris[name_filled] = filled
                # self.addShape(filled)

    def make_delaunny(self):
        for i,contour_dicts in enumerate(self.per_contour_triangles.values()):
            triangle_keys = list(contour_dicts.keys())
            triangle_keys_next = np.concatenate((triangle_keys , list(self.per_contour_triangles[f"contour{i+1}"].keys())) )
            triangle_values = list(contour_dicts.values())
            triangle_values_next = np.concatenate((triangle_values , list(self.per_contour_triangles[f"contour{i+1}"].values())) )
            curr_next_triangles = dict(zip(triangle_keys_next,triangle_values_next))
            # print(curr_next_triangles)

            c=0
            for i, key in enumerate(triangle_keys):             
                triangle = self.triangles[key]
                is_delaunny = not findViolations(self.triangles, triangle)
                if(is_delaunny): pass
                    # self.new_triangles[key] = triangle 
                    # triangle.color = Color.RED
                    # triangle.width =0.3

                    # self.addShape(triangle)
                else: 
                    new_key1, new_tri1, new_key2, new_tri2 = if_not_delaunny(triangle, curr_next_triangles)
                    self.new_triangles[new_key1] = new_tri1
                    self.new_triangles[new_key2] = new_tri2
                    new_tri1.width =0.3
                    new_tri2.width =0.3

                    # self.addShape(new_tri1)
                    # self.addShape(new_tri2)
        for key in self.triangles.keys():
            self.removeShape(key)
        self.triangles = {}
        for keys, new_triangle in self.new_triangles.items():
            # print("new triangle ", new_triangle)
            self.addShape(new_triangle,keys)
        print("end of loop")


    def peak(self, inner):
        inner_num = len(inner.points)
        x_coord = 0
        y_coord = 0
        peak_triangles ={}
        # Compute centroid
        for point2d in inner:
            x_coord += point2d.x
            y_coord += point2d.y
        average = (x_coord / inner_num, y_coord / inner_num)
        average_point = Point2D(average, color=Color.ORANGE, size=0.5)
        # self.addShape(average_point)

        # Fan triangulation
        for i in range(len(inner)):
            point_current = inner[i]
            point_next = inner[(i + 1) % len(inner)]  # wrap around to close the loop

            tr = Triangle2D(point_current, point_next, average_point, width=0.3, color=Color.YELLOW)
            name = str(random())
            peak_triangles[name] = tr
            # self.addShape(tr, name)
        # contour_idx = f"contour{len(self.per_contour_triangles) + 1}"
        # self.per_contour_triangles[contour_idx] = peak_triangles

        
    def task2(self):
        
        curve_points = self.task1()
        print("size of contours:",len(curve_points))  

        for i in range(len(curve_points)-1):
            # print(i)

            if (i==0):
                self.peak(curve_points[i])
            self.triangulate_between(curve_points[i],curve_points[i+1])

        # self.check_delaunay_status()

        # print("number of all triangles :", len(self.all_triangles))
        # return self.triangles
    
    def task2a(self):
        self.make_delaunny()
    
    def task2b(self):
        self.task2()
        all_triangles_3d = []
        
        print(list(self.per_contour_triangles.values()))
        for contour_index, contour_triangles in enumerate(list(self.per_contour_triangles.values())):            
            contour_triangles = contour_triangles.values()
            print(f"{contour_index}", contour_triangles)
            z = contour_index * H_STEP
            print(contour_triangles)


            for i,tri in enumerate(contour_triangles):
                t3d_name = str(random())
                l1_key = f"l1_{i}"
                l2_key = f"l2_{i}"
                l3_key = f"l3_{i}"

                print(tri)
                if (i%2 == 0):
                    h1 = 0
                    h2 = H_STEP
                else:
                    h1 = H_STEP
                    h2 = 0
                point3d =[(tri.x1, tri.y1, z + h2),
                    (tri.x2, tri.y2, z + h2),
                    (tri.x3, tri.y3, z + h1)]
                # print(point3d)
                line_color = COLORS[contour_index]  # Clamp to available colors

                t3d = PointSet3D(points = point3d,size = 1, color = Color.BLACK)
                line1_3d = Line3D((tri.x1, tri.y1, z + h2),(tri.x2, tri.y2, z + h2),  width=0.5, color=line_color) 
                line2_3d = Line3D((tri.x2, tri.y2, z + h2),(tri.x3, tri.y3, z + h1),  width=0.5, color=line_color) 
                line3_3d = Line3D((tri.x1, tri.y1, z + h2),(tri.x3, tri.y3, z + h1),  width=0.5, color=line_color) 

                # contour_3d.append(t3d)
                
                self.addShape(t3d,t3d_name)
                self.addShape(line1_3d)
                self.addShape(line2_3d)
                self.addShape(line3_3d)





                # self.addShape(triangle3d)

            # all_triangles_3d.append(contour_3d)

        return all_triangles_3d


    def showViolations(self,tri:Triangle2D):
        c = tri.getCircumCircle()
        c.color = Color.YELLOW
        c.width = 0.3
        self.addShape(c, f"vc{self.violations}")

        self.violations += 1




if __name__ == "__main__":
    app = Terrain()
    app.mainLoop()


