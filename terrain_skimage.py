import cv2
import numpy as np
from vvrpywork.constants import Key, Mouse, Color
from vvrpywork.scene import Scene2D
from vvrpywork.shapes import Point2D, PointSet2D, LineSet2D,Polygon2D, Line2D
from contour_map_function import generate_contour_function, generate_contour_image
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

WIDTH = 750
HEIGHT = 740
COLORS = [Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW, Color.ORANGE, Color.MAGENTA, Color.YELLOWGREEN, Color.CYAN]
class Terrain(Scene2D):
    def __init__(self):
        super().__init__(WIDTH, HEIGHT, "Terrain")
        # self.run_task_1 =False
        # self.run_task_1_polygon = False

        # if self.run_task_1:
        self.task1()  # Initialize the scene

    def task1(self):
        self.filename = generate_contour_image()

        # Load image in grayscale
        self.img = cv2.imread("contour_maps/blurred2.png")
        self.img = cv2.imread(self.filename)



        # ***************** DETECT CONTOURS **********************
        self.lower_black = np.array([0, 0, 0])
        self.upper_black = np.array([50, 50, 50])
        self.mask = cv2.inRange(self.img, self.lower_black, self.upper_black)
        # Convert to boolean and skeletonize
        self.skeleton = skeletonize(self.mask > 0)
        self.skeleton_uint8 = (self.skeleton * 255).astype(np.uint8)
        # Find contours
        self.contours, _ = cv2.findContours(self.skeleton_uint8, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        self.half_contours = self.contours[::2]
        # names = []
        self.count = 0
        self.count_points = 0
        # *************** DRAW IMAGE *********************
        # Image shape for normalization
        w,h = self.img.shape[1], self.img.shape[0]
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
        for contour in self.half_contours:
            self.pointset = PointSet2D(size=0.5)
            self.sparse_pointset= PointSet2D(size=1)
            self.lineset = LineSet2D(width = 0.5)
            nx_minus1, ny_minus1 = None, None
            for pt in contour:
                x_pixel, y_pixel = pt[0]
                # Normalize pixel coords to [-1, 1]
                nx = 2 * (x_pixel / w) - 1
                ny = 2 * (y_pixel / h) - 1
                pn = Point2D([nx,ny])
                self.pointset.add(pn)
                if nx_minus1 is not None and ny_minus1 is not None: 
                # if self.count_points%40 !=0:
                    pn_minus1 = Point2D([nx_minus1,ny_minus1],color=Color.RED)
                    self.lineset.add(Line2D(pn,pn_minus1,color=Color.BLACK))

                    if self.count_points%5 ==0:
                        self.sparse_pointset.add(pn_minus1)
                nx_minus1, ny_minus1 = nx, ny
                self.count_points +=1 
            print(f"number of points in {self.count}th contour:",len(self.sparse_pointset.points))
            self.name = f"contour_{self.count}"
            self.addShape(self.lineset)
            self.addShape(self.sparse_pointset, self.name)
            
            self.polygon = Polygon2D(self.sparse_pointset, width =0.5,reorderIfNecessary = False,color=Color.RED)
            self.addShape(self.polygon) 
            # names.append(self.name)

            
            self.count += 1

        # self.names = names  # store for later if needed

    

if __name__ == "__main__":
    app = Terrain()
    app.mainLoop()


