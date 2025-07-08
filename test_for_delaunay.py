import cv2
import numpy as np
from vvrpywork.constants import Key, Mouse, Color
from vvrpywork.scene import Scene2D, Scene3D
from vvrpywork.shapes import Point2D, PointSet2D, LineSet2D,Polygon2D, Line2D, Triangle2D,Point3D, PointSet3D,LineSet3D,Line3D
from contour_map_function import generate_contour_function, generate_contour_image
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from random import random
from helper_functions import findViolations, if_not_delaunny,findAdjacentTriangle,findAdjacentTriangle3D
from delaunay import delaunay_bowyer_watson, circumcircle_contains
import math
N =15
WIDTH = 950
HEIGHT = 950
COLORS = [Color.RED, Color.MAGENTA,  Color.ORANGE, Color.YELLOW, Color.GREEN, Color.DARKGREEN, Color.BLUE, Color.BLACK, Color.CYAN,Color.GRAY , Color.WHITE]
H_STEP = 0.1
MIN_ANGLE = 20 # degrees
MAX_AREA = 100  # adjust as needed
class Terrain(Scene2D):
    
    def __init__(self):
        super().__init__(WIDTH, HEIGHT, "Terrain")
        # self.set_slider_value(0, H_STEP)
        self.lineset = LineSet2D(width=0.5,color=Color.RED)
        self.pointset = PointSet2D(size=0.5,color=Color.RED)

        self.triangles:dict[str, Triangle2D] = {}
        self.per_contour_triangles = {}
        self.names = []
        self.half_contours = []
       
        self.violations = 0

        self.pointlist2d = []
        self.new_triangles:dict[str, Triangle2D] = {}
        self.filled_tris:dict[str, Triangle2D] = {} 

        # self.delaunay()
    def on_key_press(self, symbol, modifiers):
        if symbol == Key._1:
            self.plot_task1()

        if symbol == Key._2:
            # self.plot_triangles()
            # self.delaunay()
            self.plot_delaunay()
        if symbol == Key._3:
            self.lift_mesh()
        if symbol == Key._4:
            self.dual_graph()       

        if symbol == Key.D:
            self.check_delaunay_status()
        
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
            for key, value in self.per_contour_triangles.items():
                self.removeShape(key)
            self.filled_tris = {}

        # if symbol == Key._4:
        #     self.task2b()

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


    def task1(self):
        filename = generate_contour_image()

        # img = cv2.imread("contour_maps/pain2t.png")
        # img = cv2.imread("contour_maps/paint3.png")
        # img = cv2.imread("contour_maps/blurred2.png")
        # img = cv2.imread("contour_maps/bandw.png")
        img = cv2.imread("contour_maps/to_draw.png")
        # img = cv2.imread("contour_maps/four_contour.png")

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
            rcontour = self.resample_contour(c, N *(i+1))
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
    

    def triangle_area(self, p1, p2, p3):
        # Shoelace formula
        return abs(0.5 * (p1[0]*(p2[1]-p3[1]) +
                        p2[0]*(p3[1]-p1[1]) +
                        p3[0]*(p1[1]-p2[1])))
    
    def triangle_angles(self, p1, p2, p3):
        def angle(a, b, c):  # angle at point b
            ab = (a[0] - b[0], a[1] - b[1])
            cb = (c[0] - b[0], c[1] - b[1])
            dot = ab[0]*cb[0] + ab[1]*cb[1]
            norm_ab = math.hypot(*ab)
            norm_cb = math.hypot(*cb)
            cos_theta = dot / (norm_ab * norm_cb)
            cos_theta = min(1.0, max(-1.0, cos_theta))  # Clamp due to numerical errors
            return math.degrees(math.acos(cos_theta))

        A = angle(p2, p1, p3)
        B = angle(p1, p2, p3)
        C = 180.0 - A - B
        return A, B, C
    

    def delaunay(self):
        pointlist2d = self.task1()  # list of contours with .points
        all_points = []             # all 2D points (for triangulation)
        point_to_contour = {}       # maps point tuple -> contour index
        contour_ids = []
        outer_contour = []
        for outer_points in pointlist2d[-1].points:
            outer_contour.append(tuple(outer_points))
            # print(outer_contour)

        # Step 1: Collect all points and remember ownership
        for contour_idx, contour in enumerate(pointlist2d):
            contour_id = f"contour{contour_idx + 1}"
            contour_ids.append(contour_id)

            for pt in contour.points:
                pt_tuple = (pt[0], pt[1])
                all_points.append(pt_tuple)
                point_to_contour[pt_tuple] = contour_id

        print(f"Total input points: {len(all_points)}")

        # Step 2: Run Delaunay on all points
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
                                # Before adding triangle
                

                A, B, C = self.triangle_angles(p1t, p2t, p3t)
                area = self.triangle_area(p1t, p2t, p3t)
                print(area)

                if min(A, B, C) < MIN_ANGLE:
                    continue  # Reject triangle with bad angle

                if area > MAX_AREA:
                    continue  # Reject large triangle

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
            name = str(random())
            self.triangles[name] = tr
            peak_triangles[name] = tr

        contour_idx = "contour1"
        self.per_contour_triangles[contour_idx] = peak_triangles

    def plot_delaunay(self):

        self.delaunay()
        self.peak(self.pointlist2d[0])
        for contour_idx, contour in enumerate(self.per_contour_triangles):
            for name,tri in self.per_contour_triangles[contour].items():
                self.addShape(tri, name)

  
    def plot_task1(self):
        self.pointlist2d = self.task1()
        self.addShape(self.lineset)

        for i in range(len(self.half_contours)):
            self.addShape(self.pointlist2d[i], self.names[i])

    
    def polygon_area(self, polygon):
        # Shoelace formula (returns absolute area)
        area = 0
        n = len(polygon)
        for i in range(n):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % n]
            area += (x1 * y2 - x2 * y1)
        return abs(area) / 2
    
    def lift_mesh(self):
        
        self.delaunay()
        self.peak(self.pointlist2d[0])
        # STEP 1: Build 2D → 3D point mapping
        point2d_to_3d = dict()
        for i, contour in enumerate(self.pointlist2d):  # Skip last if it's a duplicate
            z = (i + 1) * H_STEP
            if(i!=len(self.pointlist2d)-1):
                for pt in contour:
                    key = (pt.x, pt.y)
                    point2d_to_3d[key] = (pt.x, pt.y, z)
                    p3d = Point3D((pt.x, pt.y, z),size = 0.4, color = Color.RED)
                    self.addShape(p3d, str(random()))
                    
            else: 
                point2d_to_3d[(contour.x, contour.y)] = (contour.x, contour.y, 0)
                p3d = Point3D((contour.x, contour.y, 0),size = 0.4, color = Color.GREEN)
                self.addShape(p3d, str(random()))
            

        all_3d_contours =[]
        all_3d_point_names =[]
        # STEP 2: Loop through triangulation
        for contour_index, (contour_name, triangles) in enumerate(self.per_contour_triangles.items()):
            color = COLORS[contour_index]

            contour_3d_name =[]
            contour_3d=[]
            for tri_name, tri in triangles.items():
                key = str(random())
                p1_2d = (tri.x1, tri.y1)
                p2_2d = (tri.x2, tri.y2)
                p3_2d = (tri.x3, tri.y3)

                # Lookup 3D versions
                p1_3d = point2d_to_3d.get(p1_2d)
                p2_3d = point2d_to_3d.get(p2_2d)
                p3_3d = point2d_to_3d.get(p3_2d)

                tri3d =[p1_3d,p2_3d,p3_3d]
                # print(point3d)

                t3d = PointSet3D(points = tri3d ,size = 1, color = Color.BLACK)
                contour_3d.append(t3d)
                contour_3d_name.append( key)
                

                # Draw triangle edges in 3D
                self.addShape(Line3D(p1_3d, p2_3d, resolution=3, width=0.5, color=color))
                self.addShape(Line3D(p2_3d, p3_3d, resolution=3, width=0.5, color=color))
                self.addShape(Line3D(p1_3d, p3_3d, resolution=3, width=0.5, color=color))

            all_3d_point_names.append(contour_3d_name)
            all_3d_contours.append(contour_3d)

        return all_3d_contours, all_3d_point_names

    def dual_graph(self):
        all_3d_points, all_3d_point_names = self.lift_mesh()

        centroids = {}  # Store centroids by triangle key
        lines = []      # Store lines between centroids
        # Step 1: Calculate centroids for all triangles
        for i in range(len(all_3d_points)):    
            for j, point3d in enumerate(all_3d_points[i]):
                key = str(random())
                A = point3d[0]
                B = point3d[1]
                C = point3d[2]

                centroid_x = (A.x + B.x + C.x) / 3
                centroid_y = (A.y + B.y + C.y) / 3
                centroid_z = (A.z + B.z + C.z) / 3

                centroid = Point3D((centroid_x, centroid_y, centroid_z),size =0.4,color=(1, 0, 0))
                centroids[all_3d_point_names[i][j]] = centroid # dictionary of Point3D ' s
                self.addShape(centroid,key)  # Visualize centroid points

        print("endo of loop")
        for i in range(len(all_3d_points)):    
            for j, point3d in enumerate(all_3d_points[i]):
                # key = str(random())
                A = point3d[0]
                B = point3d[1]
                C = point3d[2]
                
                for edge in [(A, B), (B, C), (C, A)]:
                    adj_key, adj_xy = findAdjacentTriangle3D(all_3d_points, all_3d_point_names, edge[0], edge[1], all_3d_point_names[i][j])
                    if adj_key and (adj_key in centroids):
                        if ((adj_key),all_3d_point_names[i][j]) not in lines: 
                            
                            line = Line3D(centroids[all_3d_point_names[i][j]], centroids[adj_key] ,width =0.5,color=Color.BLACK)
                            lines.append(((adj_key),all_3d_point_names[i][j]))
                            self.addShape(line) 


    def showViolations(self,tri:Triangle2D):
        c = tri.getCircumCircle()
        c.color = Color.YELLOW
        c.width = 0.3
        self.addShape(c, f"vc{self.violations}")

        self.violations += 1




if __name__ == "__main__":
    app = Terrain()
    app.mainLoop()


