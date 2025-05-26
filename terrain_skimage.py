import cv2
import numpy as np
from vvrpywork.constants import Key, Mouse, Color
from vvrpywork.scene import Scene2D
from vvrpywork.shapes import Point2D, PointSet2D, LineSet2D,Polygon2D, Line2D, Triangle2D
from contour_map_function import generate_contour_function, generate_contour_image
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from scipy.spatial import Delaunay

WIDTH = 750
HEIGHT = 740
COLORS = [Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW, Color.ORANGE, Color.MAGENTA, Color.YELLOWGREEN, Color.CYAN]
class Terrain(Scene2D):
    def __init__(self):
        super().__init__(WIDTH, HEIGHT, "Terrain")
        self.triangles:dict[str, Triangle2D] = {}
        self.sparse_pointlist = []
        self.reset()

    def reset(self):
        self.run_task_1 =False
        self.run_task_1_polygon = False

    def on_key_press(self, symbol, modifiers):
        if symbol == Key._1:
            self.task1()

            if symbol == Key._2:
                self.task2()
                


    def task1(self):
        filename = generate_contour_image()

        # Load image in grayscale
        img = cv2.imread("contour_maps/blurred2.png")
        # self.img = cv2.imread(self.filename)



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
        # points_with_colors = PointSet2D()
        # for x in range(w):
        #     for y in range(h):
        #         # Get color at pixel (y,x) - remember OpenCV uses BGR and img[y,x]
        #         b, g, r = self.img[y, x]
                
        #         # Normalize pixel coordinates to [-1, 1]
        #         nx = (x / w) * 2 - 1
        #         ny = (y / h) * 2 - 1
                
        #         # Store as tuple: (normalized coords, color as RGB)
        #         points_with_colors.add(Point2D((nx, ny),color = (b,g,r)))
        # self.addShape(points_with_colors)

        # *************** DRAW CONTOURS *****************
        
        for contour in half_contours:
            pointset = PointSet2D(size=0.5)
            sparse_pointset= PointSet2D(size=1)
            lineset = LineSet2D(width = 0.5)
            nx_minus1, ny_minus1 = None, None
            for pt in contour:
                x_pixel, y_pixel = pt[0]
                # Normalize pixel coords to [-1, 1]
                nx = 2 * (x_pixel / w) - 1
                ny = 2 * (y_pixel / h) - 1
                pn = Point2D([nx,ny])
                pointset.add(pn)
                if nx_minus1 is not None and ny_minus1 is not None: 
                    pn_minus1 = Point2D([nx_minus1,ny_minus1],color=Color.RED)
                    lineset.add(Line2D(pn,pn_minus1,color=Color.BLACK))

                    if count_points% 10 ==0:
                        sparse_pointset.add(pn_minus1)
                nx_minus1, ny_minus1 = nx, ny
                count_points +=1 

            self.sparse_pointlist.append(pointset)
            print(f"number of points in {count}th contour:",len(sparse_pointset.points))
            name = f"contour_{count}"
            self.addShape(lineset)
            self.addShape(sparse_pointset, name)
            # self.polygon = Polygon2D(self.sparse_pointset, width =0.5,reorderIfNecessary = False,color=Color.RED)
            # self.addShape(self.polygon) 
            # names.append(self.name)

            
            count += 1
        return self.sparse_pointlist
        # self.names = names  # store for later if needed

    def task2(self):
        lineset = LineSet2D(width = 0.5)

        self.curve_points = self.task1()
        A = self.curve_points[0]
        B = self.curve_points[1]
        pointsAB = np.concatenate([A.points,B.points])
        for i, p in enumerate(pointsAB):
            lineset.add(Line2D())


        self.addShape(self.triangles)


if __name__ == "__main__":
    app = Terrain()
    app.mainLoop()


