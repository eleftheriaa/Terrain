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
N =15
WIDTH = 950
HEIGHT = 950
# Color.MAGENTA,  Color.ORANGE,  Color.DARKGREEN, Color.BLACK, 
COLORS =[ Color.DARKRED, Color.RED, Color.ORANGE, Color.YELLOW, Color.GREEN, Color.DARKGREEN, Color.BLUE,Color.CYAN, Color.GRAY , Color.WHITE]
H_STEP = 0.1
MIN_ANGLE = 11 # degrees
MAX_AREA = 0.08  # adjust as needed



class Terrain_3D(Scene3D):
    
    def __init__(self, terrain2d ):
        super().__init__(WIDTH, HEIGHT, "Terrain 3D", n_sliders=1)

        self.set_slider_value(0, H_STEP) # uncomment for Scene3D
        self.terrain2d = terrain2d
        self.pointlist2d = self.terrain2d.pointlist2d
        self.per_contour_triangles = self.terrain2d.per_contour_triangles
        self.d_flag =self.terrain2d.d_flag

        self.do_not_meet_criteria =[]
        self.new_points = []
        self.new_points_clustered =[]
        self.l1_names = []
        self.l2_names = []
        self.l3_names = []
        self.point3d_dict = []
        self.printHelp()

# uncomment for Scene3D
    def on_slider_change(self, slider_id, value):
        if slider_id == 0:
            self.point_speed = value * 0.01
            print("value:", value)
            self.h_step = value
            self.flag = 2
        else: 
            self.flag =1

    def printHelp(self):
        self.print(f"\
3: Show Terrain\n\
4: Dual Graph\n\
5: Min distance between random points (dual graph)\n\
6: like 5 + slope < 10%\n")

    def on_key_press(self, symbol, modifiers):
        if symbol == Key._3 and self.flag == 2:
            for i in range(len(self.point3d_dict)):
                self.removeShape(self.point3d_dict[i])
            for i in range(len(self.l1_names)):
                self.removeShape(self.l1_names[i])
                self.removeShape(self.l2_names[i])
                self.removeShape(self.l3_names[i])
            if(self.d_flag == 1): self.pointlist2d = self.terrain2d.task1(N)
            elif(self.d_flag ==2): self.pointlist2d = self.terrain2d.task1(40)


            self.lift_mesh(self.h_step)
        
        if symbol == Key._4 :
            if(self.d_flag==1):self.pointlist2d = self.terrain2d.task1(N)
            elif(self.d_flag==2):self.pointlist2d = self.terrain2d.task1(40)

            print("type",type(self.h_step))
            self.triangle_keys = self.dual_graph(self.h_step)     

        if symbol == Key._5:
            self.start = random.choice(self.triangle_keys)
            self.goal = random.choice([k for k in self.triangle_keys if k != self.start])

            self.path_finding(self.start, self.goal)   

        if symbol == Key._6:
            self.path_finding_with_slope_limit(self.start, self.goal)
            
    
    def lift_mesh(self, h_step):
        self.point3d_dict =[]
        if(self.d_flag == 1):
            self.terrain2d.delaunay()
            self.terrain2d.peak(self.pointlist2d[0])

        elif(self.d_flag ==2 ): 
            self.terrain2d.peak(self.pointlist2d[0])
            # self.terrain2d.task2(N=40)
            # for i in range(len(self.pointlist2d) - 1):
            #     A = self.pointlist2d[i]     # inner
            #     B = self.pointlist2d[i + 1] # outer
            #     self.terrain2d.triangulate_between(A, B)
            
        print(self.pointlist2d)
        print("\n*****\nedw einai ta kala trigwnakia\n",self.per_contour_triangles)
        for i, contour in enumerate(self.pointlist2d):  # Skip last if it's a duplicate
                print(contour)
        # STEP 1: Build 2D → 3D point mapping
        point2d_to_3d = dict()
        colors =[]
        for i, contour in enumerate(self.pointlist2d):  # Skip last if it's a duplicate
            print(i)
            z = (i + 1) * h_step
            colors.append(COLORS[i])
            if(i!=len(self.pointlist2d)-1 or self.d_flag == 2 ):
                if isinstance(contour, PointSet2D):
                    for pt in contour:
                        key_point = str(random.random())
                        self.point3d_dict.append(key_point)
                        key = (pt.x, pt.y)
                        point2d_to_3d[key] = (pt.x, pt.y, z)
                        p3d = Point3D((pt.x, pt.y, z),size =0.5, color = COLORS[i])
                        self.addShape(p3d,key_point)
                else:
                    key_point = str(random.random())
                    self.point3d_dict.append(key_point)
                    point2d_to_3d[(contour.x, contour.y)] = (contour.x, contour.y, 0)
                    p3d = Point3D((contour.x, contour.y, 0),size =0.5, color = COLORS[i])
                    self.addShape(p3d, key_point)
                    
            else: 
                key_point = str(random.random())
                self.point3d_dict.append(key_point)
                point2d_to_3d[(contour.x, contour.y)] = (contour.x, contour.y, 0)
                p3d = Point3D((contour.x, contour.y, 0),size =0.5, color = COLORS[i])
                self.addShape(p3d, key_point)
        # print("len of poilist2d", i)
        # for key in point2d_to_3d:
        #     print("dictionary:",point2d_to_3d[key])
        all_3d_contours =[]
        all_3d_point_names =[]
        # STEP 2: Loop through triangulation
        for contour_index, (triangles) in enumerate(self.per_contour_triangles.values()):
            print("contour_index:",contour_index)
            color = COLORS[contour_index]
            print(color)
            contour_3d_name =[]
            contour_3d=[]
            for tri in triangles.values():

                key = str(random.random())
                p1_2d = (tri.x1, tri.y1)
                p2_2d = (tri.x2, tri.y2)
                p3_2d = (tri.x3, tri.y3)

                # Lookup 3D versions
                p1_3d = point2d_to_3d.get(p1_2d)
                p2_3d = point2d_to_3d.get(p2_2d)
                p3_3d = point2d_to_3d.get(p3_2d)

                tri3d =[p1_3d,p2_3d,p3_3d]
                if None in (p1_3d, p2_3d, p3_3d):
                    print(f"Skipping triangle due to missing 3D point: {p1_3d}, {p2_3d}, {p3_3d}")
                    continue
                # print(point3d)

                t3d = PointSet3D(points = tri3d ,size = 1, color = Color.BLACK)
                contour_3d.append(t3d)
                contour_3d_name.append( key)
                
                l1_name = str(random.random())
                l2_name = str(random.random())
                l3_name = str(random.random())

                self.l1_names.append(l1_name)
                self.l2_names.append(l2_name)
                self.l3_names.append(l3_name)

                # Draw triangle edges in 3D
                self.addShape(Line3D(p1_3d, p2_3d, resolution=3, width=1, color=color),l1_name)
                self.addShape(Line3D(p2_3d, p3_3d, resolution=3, width=1, color=color),l2_name)
                self.addShape(Line3D(p1_3d, p3_3d, resolution=3, width=1, color=color), l3_name)

            all_3d_point_names.append(contour_3d_name)
            all_3d_contours.append(contour_3d)
        print("len of per contour ...", contour_index)

        return all_3d_contours, all_3d_point_names

    def dual_graph(self, h_step):
        self.all_3d_points, self.all_3d_point_names = self.lift_mesh(h_step)

        self.dual_centroids = {}   # triangle name -> Point3D
        self.dual_graph_dict = {}       # triangle name -> list of (neighbor, distance)
        self.line_names_dg = []
        lines = set()      # Store lines between centroids
        # Step 1: Calculate centroids for all triangles
        for i in range(len(self.all_3d_points)):    
            for j, point3d in enumerate(self.all_3d_points[i]):
                key_raw = self.all_3d_point_names[i][j]
                key = str(key_raw)
                A = point3d[0]
                B = point3d[1]
                C = point3d[2]

                centroid_x = (A.x + B.x + C.x) / 3
                centroid_y = (A.y + B.y + C.y) / 3
                centroid_z = (A.z + B.z + C.z) / 3

                centroid = Point3D((centroid_x, centroid_y, centroid_z),size =0.5,color=Color.BLACK)
                self.dual_centroids[self.all_3d_point_names[i][j]] = centroid # dictionary of Point3D ' s
                self.dual_graph_dict[key] = []
                self.addShape(centroid,key)  # Visualize centroid points

        print("endo of loop")
        for i in range(len(self.all_3d_points)):    
            for j, point3d in enumerate(self.all_3d_points[i]):
                key = self.all_3d_point_names[i][j]

                # key = str(random())
                A = point3d[0]
                B = point3d[1]
                C = point3d[2]
                
                for edge in [(A, B), (B, C), (C, A)]:
                    adj_key_raw, adj_xy = findAdjacentTriangle3D(self.all_3d_points, self.all_3d_point_names, edge[0], edge[1], self.all_3d_point_names[i][j])
                    adj_key = str(adj_key_raw)
                    if adj_key_raw  and (adj_key in self.dual_centroids):
                        pair = tuple(sorted([ self.all_3d_point_names[i][j], adj_key]))
                        if pair not in lines: 
                            line_name_dg = str(random.random())
                            c1 = self.dual_centroids[self.all_3d_point_names[i][j]]
                            c2 = self.dual_centroids[adj_key]
                            dist = ((c1.x - c2.x) ** 2 + (c1.y - c2.y) ** 2 + (c1.z - c2.z) ** 2) ** 0.5

                            line = Line3D(c1,c2  ,width =1,color=Color.BLACK)
                            self.addShape(line, line_name_dg) 
                            self.line_names_dg.append(line_name_dg)
                            lines.add(pair)
                            # lines.append(((adj_key),all_3d_point_names[i][j]))
                            self.dual_graph_dict[key].append((adj_key, dist))
                            self.dual_graph_dict[adj_key].append((self.all_3d_point_names[i][j], dist))


        return list(self.dual_graph_dict.keys())  # for picking random start/goal
    


    def scale_mesh_xy_for_slope(self, max_slope=0.1):
        from itertools import combinations
        for line in range(len(self.line_names_dg)):
            self.removeShape(self.line_names_dg[line])
        for k in self.dual_graph_dict.keys():
            self.removeShape(k)
        # Step 1: Flatten all triangle centroids
        centroids = []
        for triangle_group in self.all_3d_points:
            for triangle in triangle_group:
                A, B, C = triangle
                cx = (A.x + B.x + C.x) / 3
                cy = (A.y + B.y + C.y) / 3
                cz = (A.z + B.z + C.z) / 3
                centroids.append((cx, cy, cz))

        # Step 2: Estimate maximum vertical step (dz) between triangle centroids
        max_dz = 0
        for c1, c2 in combinations(centroids, 2):
            dz = abs(c1[2] - c2[2])
            max_dz = max(max_dz, dz)

        # Step 3: Compute minimum horizontal distance between centroids
        min_horizontal = float("inf")
        for c1, c2 in combinations(centroids, 2):
            dx = c1[0] - c2[0]
            dy = c1[1] - c2[1]
            dist_xy = (dx**2 + dy**2)**0.5
            if dist_xy > 0:
                min_horizontal = min(min_horizontal, dist_xy)

        # Step 4: Compute required min horizontal to keep slope < max_slope
        required_horizontal = max_dz / max_slope if max_slope > 0 else float('inf')

        if min_horizontal == 0:
            print("Error: Flat or overlapping mesh, cannot scale.")
            return

        scale_factor = required_horizontal / min_horizontal
        print(f"Scaling (x,y) of all 3D points by {scale_factor:.3f} to ensure slope < {max_slope}")

        # Step 5: Apply scale to all 3D points (x and y only)
        scaling  = 20
        for i,triangle_group in enumerate(self.all_3d_points):
            print("tirangle group ", triangle_group)
            for j,triangle in enumerate(triangle_group):
                for p in triangle:
                    p.x *= scaling
                    p.y *= scaling
                    self.updateShape(self.all_3d_point_names[i][j])
        print("its done")
        






    def path_finding_with_slope_limit(self, start_key, goal_key, slope_threshold=0.60):

        self.scale_mesh_xy_for_slope()
        # self.dual_graph(self.h_step)
        # self.path_finding(start_key, goal_key)
        # restricted_graph = {}
        # i=0
        # for key, neighbors in self.dual_graph_dict.items():
        #     restricted_graph[key] = []
        #     for neighbor_key, distance in neighbors:

        #         # Get Point3D centroids
        #         c1 = self.dual_centroids[key]
        #         c2 = self.dual_centroids[neighbor_key]

        #         dz = abs(c1.z - c2.z)
        #         dx = c1.x - c2.x
        #         dy = c1.y - c2.y
        #         horizontal_dist = (dx**2 + dy**2) ** 0.5
        #         # print(horizontal_dist)
        #         # print(dz)
        #         # print("- * -")
        #         if horizontal_dist == 0:  # Avoid division by zero
        #             continue

        #         slope = dz / horizontal_dist
        #         print("slope value:", slope)

        #         if slope <= slope_threshold:
        #             restricted_graph[key].append((neighbor_key, distance))
        #             # self.addShape(Point3D((p.x, p.y, p.z), size=0.7, color=Color.GREEN))

        #         else:
        #             # Optionally print rejected edge
        #             # print(f"Skipped edge {key} -> {neighbor_key} due to slope {slope:.2f}")
        #             pass

        # # Now run dijkstra on restricted graph
        # path, total_cost = dijkstra(restricted_graph, start_key, goal_key)

        # # Visualize path if desired
        # if path:
        #     for i in range(len(path)-1):
        #         a = self.dual_centroids[path[i]]
        #         b = self.dual_centroids[path[i+1]]
        #         line = Line3D(a, b, width=1.2, color=Color.GRAY)
        #         self.addShape(line)
        # else:
        #     print(" No valid path found with slope restriction.")

        # return path, total_cost

    def path_finding(self, start_key, goal_key):
        if not hasattr(self, 'dual_graph_dict') or not hasattr(self, 'dual_centroids'):
            print("Dual graph not built yet. Call dual_graph() first.")
            return

        path, total_cost = dijkstra(self.dual_graph_dict, start_key, goal_key)
        print(f"Shortest path cost: {total_cost:.3f}, path: {path}")

        goal_point = self.dual_centroids[goal_key]
        self.addShape(Point3D((goal_point.x, goal_point.y, goal_point.z), size=0.8, color=Color.RED))
        for key in path:
            p = self.dual_centroids[key]
            self.addShape(Point3D((p.x, p.y, p.z), size=0.7, color=Color.WHITE))

        for i in range(len(path) - 1):
            p1 = self.dual_centroids[path[i]]
            p2 = self.dual_centroids[path[i+1]]
            self.addShape(Line3D(p1, p2, width=1.2, color=Color.WHITE))