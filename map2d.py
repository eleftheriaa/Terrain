import cv2
import numpy as np
from vvrpywork.constants import Key, Mouse, Color
from vvrpywork.scene import Scene2D, Scene3D
from vvrpywork.shapes import Point2D, PointSet2D, LineSet2D,Polygon2D, Line2D, Triangle2D,Point3D, PointSet3D,LineSet3D,Line3D, Label2D
from contour_map_function import generate_contour_function, generate_contour_image
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
import random
from helper_functions import findViolations, if_not_delaunny,findAdjacentTriangle,findAdjacentTriangle3D
from delaunay import delaunay_bowyer_watson, circumcircle_contains,triangle_angles, circumcenter,barycenter ,triangle_area, Point
from path_finding_algorithms import dijkstra
import math
# N =15
# M = 40
WIDTH = 950
HEIGHT = 950
# Color.MAGENTA,  Color.ORANGE,  Color.DARKGREEN, Color.BLACK, 
COLORS =[ Color.DARKRED, Color.RED,Color.ORANGE, Color.YELLOW, Color.GREEN, Color.DARKGREEN, Color.BLUE,Color.CYAN, Color.GRAY , Color.WHITE]
H_STEP = 0.1
MIN_ANGLE = 5 # degrees
MAX_AREA = 0.1  # adjust as needed
class Terrain(Scene2D):
    
    def __init__(self):
        
        super().__init__(WIDTH, HEIGHT, "Terrain") # uncomment for Scene2D

        self.lineset = LineSet2D(width=0.5,color=Color.RED)
        self.pointset = PointSet2D(size=0.5,color=Color.RED)
        self.d_flag = 0

        self.triangles:dict[str, Triangle2D] = {}
        self.per_contour_triangles = {}
        self.names = []
        self.half_contours = []
        self.all_points =[]
        self.violations = 0
        self.flag  = 0
        self.pointlist2d = []
        self.new_triangles:dict[str, Triangle2D] = {}
        self.filled_tris:dict[str, Triangle2D] = {} 
        self.do_not_meet_criteria =[]
        self.new_points = []
        self.new_points_clustered =[]
        self.l1_names = []
        self.l2_names = []
        self.l3_names = []
        self.point3d_dict = []

# uncomment for Scene2D

        self.addShape(Label2D(Point2D((-0.7, 1)), text=" '1' : contour lines", size=12, bold=True), "label_1")

        self.addShape(Label2D(Point2D((-0.7, 0.95)), text=" 'C' : (costum) NON Delaunay triangulation ", size=12, bold=True), "label_2")
        self.addShape(Label2D(Point2D((-0.7, 0.91)), text=" '2' : flip edges ", size=12, bold=True), "label_3")

        self.addShape(Label2D(Point2D((-0.7, 0.86)), text=" 'D' : Delaunay triangulation ", size=12, bold=True), "label_4")
        self.addShape(Label2D(Point2D((-0.7, 0.82)), text= " '3' : Delaunay correction based on limitations", size=12, bold=True), "label_5")

        self.addShape(Label2D(Point2D((-0.7, 0.77)), text= " 'R' : reset", size=12, bold=True), "label_6")
        self.addShape(Label2D(Point2D((-0.7, 0.73)), text= " 'T' : 3D Scene", size=12, bold=True), "label_7")


    
    def on_key_press(self, symbol, modifiers):
        if symbol == Key._1:
            self.plot_task1(N = 15)

        if symbol == Key.C:
            for k in self.names:
                self.removeShape(k)
            self.removeShape("contour_lineset")
            # self.plot_task1(N = 40)
            self.plot_task2(N = 40)
            
        if symbol == Key._2:
            self.check_delaunay_status()
            # self.task2a()

        if symbol == Key.D:
            # self.plot_triangles()
            # self.delaunay()
            self.pointlist2d = self.task1(N=15)  # list of contours with .points
            self.plot_delaunay2()
        if symbol == Key._3:
            self.del_correction()
        

        
        if symbol == Key.T:
            from terrain3d import Terrain_3D
            terrain3d = Terrain_3D(self)
            terrain3d.mainLoop()
        if symbol == Key.R :
            self.violations = 0
            self.removeShape("contour_lineset")
            

            for k in self.names:
                self.removeShape(k)
            self.names = []
            for key in self.triangles.keys():
                self.removeShape(key)
            self.triangles = {}
            for key,value in self.filled_tris.items():
                value.filled=False
                self.removeShape(key)
            for  contour in (self.per_contour_triangles):
                for name in self.per_contour_triangles[contour].keys():
                    self.removeShape(name)
            self.filled_tris = {}
            for contour_idx, bad_tris in self.do_not_meet_criteria_with_contour.items():
                for i, bt in enumerate(bad_tris):
                    self.removeShape( f"bad_tr{contour_idx}_{i}")
                    self.removeShape(f"p1{contour_idx}_{i}")
                    self.removeShape( f"p2{contour_idx}_{i}")
                    self.removeShape( f"p3{contour_idx}_{i}")
                    

            self.triangles:dict[str, Triangle2D] = {}
            self.per_contour_triangles = {}
            self.lineset = LineSet2D(width=0.5,color=Color.RED)
            self.pointset = PointSet2D(size=0.5,color=Color.RED)
            self.pointlist2d = []

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

        return new_resampled             


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
    def task1(self, N):
        filename = generate_contour_image()

        # img = cv2.imread("contour_maps/pain2t.png")
        # img = cv2.imread("contour_maps/paint3.png")
        # img = cv2.imread("contour_maps/blurred2.png")
        # img = cv2.imread("contour_maps/bandw.png")
        # img = cv2.imread("contour_maps/to_draw.png")
        # img = cv2.imread("contour_maps/four_contour.png")
        img = cv2.imread("contour_maps/five_contours.png")
        # img = cv2.imread("contour_maps/two_slopes.png")



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
        self.half_contours = contours[::2]
        # *************** DRAW IMAGE *********************
        # Image shape for normalization
        w,h =img.shape[1], img.shape[0]


        # sort contours from innermost to outermost
        sorted_contours = self.half_contours[::-1]
        print(f"sorted contours:{len(sorted_contours)} ")
        print(f"contours:{len(self.half_contours)} ")

        # *************** DRAW CONTOURS *****************
        pointlist2d = []
        # self.pointlist2d =pointlist2d
        for i, c in enumerate(self.half_contours):
            self.pointset = PointSet2D(size=0.5,color=Color.RED)
            nx_minus1, ny_minus1 = None, None
            # *************** RESAMPLE CONTOURS *****************
            if (N== 15):rcontour = self.resample_contour(c, N *(i+1))
            else:rcontour = self.resample_contour(c, N )

            # print(f"contour_{i}", rcontour)
            # print(rcontour)
            for x_pixel, y_pixel in rcontour:
                # Normalize pixel coords to [-1, 1]
                nx = 2 * (x_pixel / w) - 1
                ny = 2 * (y_pixel / h) - 1
                pn = Point2D([nx,ny],color=Color.RED)
                self.pointset.add(pn)
                
                if nx_minus1 is not None and ny_minus1 is not None: 
                    pn_minus1 = Point2D([nx_minus1,ny_minus1],size=2,color=Color.RED)
                    self.lineset.add(Line2D(pn,pn_minus1,width = 2,color=Color.BLACK))
                nx_minus1, ny_minus1 = nx, ny
            self.names.append(f"contour_{i}")

            # self.addShape(lineset)
            # self.addShape(pointset, self.names[i])
            pointlist2d.append(self.pointset)
        # print("number of points in all contour",pointlist2d)   
        self.pointlist2d =pointlist2d
        return pointlist2d
    

    def point_in_polygon(self, point, contour):
        """Check if a 2D point is inside a polygon using ray casting."""
        x, y = point
        inside = False
        n = len(contour)
        px, py = zip(*contour)
        j = n - 1
        for i in range(n):
            if ((py[i] > y) != (py[j] > y)) and \
            (x < (px[j] - px[i]) * (y - py[i]) / (py[j] - py[i] + 1e-10) + px[i]):
                inside = not inside
            j = i
        return inside
    
    
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
                name = str(random.random())
                inner_tr_dict[name] = tri
                self.triangles[name] = tri
                # self.addShape(tri, name)

            
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
                filled = Triangle2D(triangle.getPoint1(), triangle.getPoint2(), triangle.getPoint3(),width = 0.5, color=Color.DARKGREEN, filled=True)
                name_filled = str(random.random()) 
                self.filled_tris[name_filled] = filled
                self.addShape(filled, name_filled)
                # self.showViolations(triangle)


    def flip_edges(self):
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
            new_triangle.width =0.5
            self.updateShape(keys)
        print("end of loop")



    def delaunay(self):
        self.d_flag = 1
        all_points = []             # all 2D points (for triangulation)
        point_to_contour = {}       # maps point tuple -> contour index
        contour_ids = []
        outer_contour = []
        # print(self.pointlist2d)
        for outer_points in self.pointlist2d[-1].points:
            outer_contour.append(tuple(outer_points))
            # print(outer_contour)

        # Step 1: Collect all points and remember ownership
        for contour_idx, contour in enumerate(self.pointlist2d):
            contour_id = f"contour{contour_idx + 1}"
            contour_ids.append(contour_id)

            for pt in contour.points:
                pt_tuple = (pt[0], pt[1])
                all_points.append(pt_tuple)
                point_to_contour[pt_tuple] = contour_id

        print(f"Total input points: {len(all_points)}")
        self.all_points = all_points
        # Step 2: Run Delaunay on all points
        # delaunay_tris = delaunay_bowyer_watson(all_points)  # each tri: [(x1,y1), (x2,y2), (x3,y3)]
        delaunay_tris = delaunay_bowyer_watson(all_points)  # each tri: [(x1,y1), (x2,y2), (x3,y3)]

        
        # Step 3: Prepare output dict
        self.per_contour_triangles = {cid: {} for cid in contour_ids}

        # Step 4: Assign triangles to contours and draw
        for i, (p1, p2, p3) in enumerate(delaunay_tris):
            p1t = (p1.x, p1.y) 
            p2t = (p2.x, p2.y) 
            p3t = (p3.x, p3.y) 

            # Calculate centroid
            cx = (p1t[0] + p2t[0] + p3t[0]) / 3
            cy = (p1t[1] + p2t[1] + p3t[1]) / 3

            # Check if inside outer contour
            if not self.point_in_polygon((cx, cy), outer_contour):
                continue  # Skip triangle outside outer contour
            else:

                contour_set = {point_to_contour.get(p, "unknown") for p in [p1t, p2t, p3t]}

                # Determine main contour: if all 3 from same contour
                if len(contour_set) == 1:
                    contour_id = contour_set.pop()
                else:
                    # Mixed triangle: assign to "mixed" or to first non-contour1
                    contour_id = next((cid for cid in contour_set if cid != "contour1"), "contour1")
                    
                # Create triangle shape
                tri = Triangle2D(p1t, p2t, p3t, width=0.3, color=Color.DARKGREEN)
                name = f"{contour_id}_tri_{i}"

                # Store triangle
                

                # Step 5: Skip rendering if triangle is fully inside contour1
                if contour_id != "contour1":
                    self.per_contour_triangles[contour_id][name] = tri
                    # self.addShape(tri, name)



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
        self.pointlist2d.append(average_point)

        # Fan triangulation
        for i in range(len(inner)):
            point_current = inner[i]
            point_next = inner[(i + 1) % len(inner)]  # wrap around to close the loop

            tr = Triangle2D(point_current, point_next, average_point, width=0.5, color=Color.YELLOW)
            name = str(random.random())
            self.triangles[name] = tr
            peak_triangles[name] = tr

        contour_idx = "contour1"
        self.per_contour_triangles[contour_idx] = peak_triangles

    def task2(self, N):
            
        self.curve_points = self.task1(N)
        print("size of contours:",len(self.curve_points))  

        for i in range(len(self.curve_points)-1):
            print(i)
            if (i==0):
                self.peak(self.curve_points[i])
            self.triangulate_between(self.curve_points[i],self.curve_points[i+1])
        self.d_flag = 2
        print("dflag",self.d_flag)


    def plot_task2(self, N):
        self.task2(N)
        for contour_idx, contour in enumerate(self.per_contour_triangles):
            print("contour_idx",contour_idx)
            for name,tri in self.per_contour_triangles[contour].items():
                self.addShape(tri, name)
        for i in range(len(self.curve_points)-1):
            print(i)
            self.addShape(self.curve_points[i], self.names[i])



    def task2a(self):
        self.flip_edges()

    
    def plot_delaunay2(self):
        self.delaunay()
        self.peak(self.pointlist2d[0])

        self.new_points = []  # Flat list of (x, y)
        self.new_points_clustered = []  # List of PointSet2D
        self.do_not_meet_criteria = []
        self.do_not_meet_criteria_with_contour = dict()

        for contour_idx, contour in enumerate(self.per_contour_triangles):
            for name, tri in self.per_contour_triangles[contour].items():
                self.addShape(tri, name)

                p1 = tri.getPoint1()
                p2 = tri.getPoint2()
                p3 = tri.getPoint3()

                tuple_tri = (Point(p1.x, p1.y), Point(p2.x, p2.y), Point(p3.x, p3.y))
                angles = triangle_angles(*tuple_tri)
                area = triangle_area(*tuple_tri)

                if min(angles) < MIN_ANGLE or area > MAX_AREA:
                    self.do_not_meet_criteria.append(tuple_tri)

                    # Store triangle under corresponding contour index
                    if contour_idx not in self.do_not_meet_criteria_with_contour:
                        self.do_not_meet_criteria_with_contour[contour_idx] = []
                    self.do_not_meet_criteria_with_contour[contour_idx].append(tuple_tri)

        print("these are the triangles that do not meet the conditions:\n", self.do_not_meet_criteria)
        self.addShape(Label2D(Point2D((0.5, 1)), text=f" MIN_ANGLE = {MIN_ANGLE} degrees\n MAX_AREA = {MAX_AREA} ", size=12, bold=True), "label")

        # # Initialize containers per contour index
        max_contours = max(self.do_not_meet_criteria_with_contour.keys(), default=-1) + 1
        print(max_contours)
        self.new_points_clustered = [PointSet2D() for _ in range(max_contours)]

        for contour_idx, bad_tris in self.do_not_meet_criteria_with_contour.items():
            for i, bad_tri in enumerate(bad_tris):
                r1, r2, r3 = bad_tri
                center = barycenter(r1, r2, r3)

                # Add to flat list
                self.new_points.append((center.x, center.y))

                # Add to corresponding contour cluster
                self.new_points_clustered[contour_idx].add(Point2D((center.x, center.y)))

                # Optionally draw
                p1 = Point2D((r1.x, r1.y), color=Color.MAGENTA)
                p2 = Point2D((r2.x, r2.y), color=Color.MAGENTA)
                p3 = Point2D((r3.x, r3.y), color=Color.MAGENTA)
                bt = Triangle2D(p1, p2, p3, filled=True, color=Color.MAGENTA)

                self.addShape(bt, f"bad_tr{contour_idx}_{i}")
                self.addShape(p1, f"p1{contour_idx}_{i}")
                self.addShape(p2, f"p2{contour_idx}_{i}")
                self.addShape(p3, f"p3{contour_idx}_{i}")
        print("clustered",self.new_points_clustered)
        print("End of loop")

            
    def del_correction(self):
        self.new_points+=self.all_points
        print(self.pointlist2d)
        print("---- * ----")
        print(self.new_points_clustered)
        self.pointlist2d = self.new_points_clustered
        for contour_idx, contour in enumerate(self.per_contour_triangles):
            for name, tri in self.per_contour_triangles[contour].items():
                self.removeShape(name)
        # self.plot_delaunay2()
        new_del_tris = delaunay_bowyer_watson(self.new_points)
        outer_contour = []
        for outer_points in self.new_points_clustered[-1].points:
            outer_contour.append(tuple(outer_points))
            # print(outer_contour)

        for i, (p1, p2, p3) in enumerate(new_del_tris):
            p1t = Point2D((p1.x, p1.y))
            p2t = Point2D((p2.x, p2.y))
            p3t = Point2D((p3.x, p3.y))

            cx = (p1t.x + p2t.x + p3t.x) / 3
            cy = (p1t.y + p2t.y + p3t.y) / 3
            if not self.point_in_polygon((cx, cy), outer_contour):
                continue 
            self.addShape(Triangle2D(p1t,p2t,p3t, width = 0.5), f"redelauany{i}")
            print("correct tri")



    def plot_task1(self, N):
        self.pointlist2d = self.task1(N)
        self.addShape(self.lineset, "contour_lineset")

        for i in range(len(self.half_contours)):
            self.addShape(self.pointlist2d[i], self.names[i])

    

if __name__ == "__main__":
    app = Terrain()
    app.mainLoop()



