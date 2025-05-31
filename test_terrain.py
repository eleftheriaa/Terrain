import cv2
import numpy as np
from vvrpywork.constants import Key, Mouse, Color
from vvrpywork.scene import Scene2D
from vvrpywork.shapes import Point2D, PointSet2D, LineSet2D,Polygon2D, Line2D, Triangle2D
from contour_map_function import generate_contour_function, generate_contour_image
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from random import random

N =50
WIDTH = 950
HEIGHT = 940
COLORS = [Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW, Color.ORANGE, Color.MAGENTA, Color.YELLOWGREEN, Color.CYAN]
class Terrain(Scene2D):
    def __init__(self):
        super().__init__(WIDTH, HEIGHT, "Terrain")
        self.triangles:dict[str, Triangle2D] = {}
        self.sparse_pointlist = []
    #     self.reset()

    # def reset(self):
    #     self.task2()

    def on_key_press(self, symbol, modifiers):
        if symbol == Key._1:
            self.task1()

        if symbol == Key._2:
            self.task2()
   

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

        img = cv2.imread("contour_maps/blurred2.png")

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
        # names = []
        count = 0
        count_points = 0
        # *************** DRAW IMAGE *********************
        # Image shape for normalization
        w,h =img.shape[1], img.shape[0]


        

        # sort contours from innermost to outermost
        sorted_contours = half_contours[::-1]
        print(f"sorted contours:{len(sorted_contours)} ")
        print(f"contours:{len(half_contours)} ")

        pointlist2d = []
        # *************** DRAW CONTOURS *****************
        for i, c in enumerate(half_contours):
            pointset = PointSet2D(size=0.5,color=Color.RED)
            lineset = LineSet2D(width = 0.5)
            nx_minus1, ny_minus1 = None, None
            # *************** RESAMPLE CONTOURS *****************
            rcontour = self.resample_contour(c, N)
            print(f"contour_{count}", rcontour)
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
            name = f"contour_{count}"
            self.addShape(lineset)
            self.addShape(pointset, name)
            pointlist2d.append(pointset)
            count += 1
        print("number of points in all contour",pointlist2d)   
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
        print(step)
        
        for i in range(0, L, step):
            print(i)
            b_curr = B[i]
            b_next = B[(i + 1) % L]
            b_next_next = B[(i + 2) % L]
            if (i%L==0):
                a_idx = self.closest_point_in_contour(b_curr,A)
                a_start = a_idx
            a_curr = A[(a_idx) % K]
            a_next = A[(a_idx + 1) % K]
            # Form 3 triangles: (a_curr, a_next, b_curr) and (b_curr, b_next, a_next)
            new_tri1 = Triangle2D(b_curr, b_next, a_curr,width=0.5,color = Color.BLUE)
            new_tri2 = Triangle2D(a_curr, a_next, b_next,width=0.5,color = Color.BLUE)

            new_tri3 = Triangle2D(b_next_next, b_next, a_next,width=0.5,color = Color.BLUE)

        
            name1 = str(random())
            name2 = str(random())
            name3 = str(random())

            self.triangles[name1] = new_tri1
            self.triangles[name2] = new_tri2
            self.triangles[name3] = new_tri3

            self.addShape(new_tri1, name1)
            self.addShape(new_tri3, name2)
            self.addShape(new_tri2, name3)
            
            a_idx +=2
            # if (a_idx == a_start+K ): break

        return self.triangles
   
    
    def task2(self):

        curve_points = self.task1()
        print("size of contours:",len(curve_points))
        
        for i in range(len(curve_points)-1):
            self.triangulate_between(curve_points[i],curve_points[i+1])







if __name__ == "__main__":
    app = Terrain()
    app.mainLoop()


