import cv2
import numpy as np
from vvrpywork.constants import Key, Mouse, Color
from vvrpywork.scene import Scene2D, Scene3D
from vvrpywork.shapes import Point2D, PointSet2D, LineSet2D,Polygon2D, Line2D, Triangle2D, PointSet3D,LineSet3D,Line3D
from contour_map_function import generate_contour_function, generate_contour_image
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from random import random
from helper_functions import findViolations, if_not_delaunny
from scipy.spatial import Delaunay

N =40
WIDTH = 950
HEIGHT = 950
COLORS = [Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW, Color.ORANGE, Color.MAGENTA, Color.YELLOWGREEN, Color.CYAN]
H_STEP = 0.1
S = Scene2D
class Terrain(S):
    
    def __init__(self):
        super().__init__(WIDTH, HEIGHT, "Terrain")
        self.triangles:dict[str, Triangle2D] = {}
        self.per_contour_triangles = {}
        self.names = []
        # self.all_triangles = []
        # self.all_triangles_keys = []
        self.violations = 0

        self.pointlist2d = []
        self.new_triangles:dict[str, Triangle2D] = {}
        self.filled_tris:dict[str, Triangle2D] = {} 


    def on_key_press(self, symbol, modifiers):
        if symbol == Key._1:
            self.task1()

        if symbol == Key._2:
            self.task2()
        
        if symbol == Key.D:
            self.check_delaunay_status()
        if symbol == Key._3:
            self.task2a()
        if symbol == Key._4:
            self.task2b()
        if symbol == Key.R :
            
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

        img = cv2.imread("contour_maps/to_draw.png")

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
            rcontour = self.resample_contour(c, N)
            print(f"contour_{i}", rcontour)
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

            self.addShape(lineset)
            self.addShape(pointset, self.names[i])
            pointlist2d.append(pointset)
            # count += 1
        # print("number of points in all contour",pointlist2d)   
        return pointlist2d
    
    def closest_point_in_contour(self,point, A):
        # Find the point in A that is closest to the given point
        min_dist = float('inf')
        closest = None
        count = 0
        for a in A:
            dist = np.linalg.norm(np.array((a.x,a.y)) - np.array((point.x,point.y)))
            if dist < min_dist:
                min_dist = dist
                closest = a
                idx = count
            count+=1
        return idx
    
    def triangulate_between(self,A, B):
        """
        A has M points, B has N points. N should divide M evenly.
        Triangulate by connecting each B[i] with the correct chunk of A.
        """
        K = len(A.points)
        L = len(B.points)
        step = 2
        # print(step)
        inner_tr_dict = {}

        for i in range(0, L, step):
            # print(i)
            b_curr = B[i]
            b_next = B[(i + 1) % L]
            b_next_next = B[(i + 2) % L]
            if (i%L==0):
                a_idx = self.closest_point_in_contour(b_curr,A)
                a_start = a_idx
            a_curr = A[(a_idx) % K]
            a_next = A[(a_idx + 1) % K]
            a_next_next = A[(a_idx + 2) % K]

            # Form 3 triangles: (a_curr, a_next, b_curr) and (b_curr, b_next, a_next)
            new_tri1 = Triangle2D(b_curr, b_next, a_curr,width=0.5,color = Color.BLACK)
            new_tri2 = Triangle2D(a_curr, a_next, b_next,width=0.5,color = Color.BLACK)

            new_tri3 = Triangle2D(b_next_next, b_next, a_next,width=0.5,color = Color.BLACK)
            new_tri4 = Triangle2D( a_next, a_next_next,b_next_next, width=0.5,color = Color.BLACK)

            for tri in [new_tri1, new_tri2, new_tri3, new_tri4]:
                name = str(random())
                inner_tr_dict[name] = tri
                self.triangles[name] = tri
                self.addShape(tri, name)

            
            a_idx +=2
        contour_id = f"contour{len(self.per_contour_triangles) + 1}"
        self.per_contour_triangles[contour_id] = inner_tr_dict
        # self.all_triangles.append(inner_tr)
        # self.all_triangles_keys.append(inner_tr_key)

    def check_delaunay_status(self) :
        if (self.triangles == {}):
            triangles = self.new_triangles
        else: triangles = self.triangles
        # Optionally reset if you're using self.names to store last-inserted triangle names
        triangle_keys = list(triangles.keys())
        c=0
        for i, key in enumerate(triangle_keys):
            triangle = triangles[key]
            is_delaunay = not findViolations(triangles, triangle)

            if (is_delaunay == True ): c +=1
            else:
                filled = Triangle2D(triangle.getPoint1(), triangle.getPoint2(), triangle.getPoint3(),width =0.5, color=Color.YELLOW, filled=True)
                name_filled = str(random()) 
                self.filled_tris[name_filled] = filled
                self.addShape(filled)
                # self.showViolations(triangle)


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
                if(is_delaunny): 
                    self.new_triangles[key] = triangle 

                    # self.addShape(triangle)
                else: 
                    new_key1, new_tri1, new_key2, new_tri2 = if_not_delaunny(triangle, curr_next_triangles)
                    self.new_triangles[new_key1] = new_tri1
                    self.new_triangles[new_key2] = new_tri2
                    
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

        # Compute centroid
        for point2d in inner:
            x_coord += point2d.x
            y_coord += point2d.y
        average = (x_coord / inner_num, y_coord / inner_num)
        average_point = Point2D(average, color=Color.ORANGE, size=0.5)
        self.addShape(average_point)

        # Fan triangulation
        for i in range(len(inner)):
            point_current = inner[i]
            point_next = inner[(i + 1) % len(inner)]  # wrap around to close the loop

            tr = Triangle2D(point_current, point_next, average_point, width=0.3, color=Color.YELLOW)
            name = str(random())
            self.addShape(tr, name)

    def task2(self):
        
        curve_points = self.task1()
        print("size of contours:",len(curve_points))  

        for i in range(len(curve_points)-1):
            print(i)

            if (i==0):
                self.peak(curve_points[i])
            self.triangulate_between(curve_points[i],curve_points[i+1])

        # self.check_delaunay_status()

        # print("number of all triangles :", len(self.all_triangles))
        
        # return self.all_triangles
    def task2a(self):
        self.make_delaunny()
    
    def task2b(self):
        all_triangles_2d = self.task2()
        all_triangles_3d = []

        for contour_index, contour_triangles in enumerate(all_triangles_2d):
            z = contour_index * H_STEP
            contour_3d = []

            for i,tri in enumerate(contour_triangles):
                if (i%2 == 0):
                    h1 = 0
                    h2 = 0.1
                else:
                    h1 = 0.1
                    h2 = 0
                point3d =((tri.x1, tri.y1, z + h2),
                    (tri.x2, tri.y2, z + h2),
                    (tri.x3, tri.y3, z + h1))
                t3d = PointSet3D(point3d,size = 1, color = Color.BLACK)
                triangle3d = LineSet3D(points=point3d,  width=1, color=Color.GREEN)
                # self.addShape(triangle3d)
                contour_3d.append(t3d)
                self.addShape(t3d)
                # self.addShape(triangle3d)

            all_triangles_3d.append(contour_3d)

        return all_triangles_3d


    def showViolations(self,tri:Triangle2D):
            c = tri.getCircumCircle()
            c.color = Color.CYAN
            c.width = 0.3
            self.addShape(c, f"vc{self.violations}")

            self.violations += 1



if __name__ == "__main__":
    app = Terrain()
    app.mainLoop()

