import cv2
import numpy as np
from vvrpywork.constants import Key, Mouse, Color
from vvrpywork.scene import Scene2D
from vvrpywork.shapes import Point2D, PointSet2D, LineSet2D
from contour_map_function import generate_contour_function, generate_contour_image
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

WIDTH = 739
HEIGHT = 744
COUNT =0
class Terrain(Scene2D):
    def __init__(self):
        super().__init__(WIDTH, HEIGHT, "Terrain")
        self.reset()  # Initialize the scene

    def reset(self):
        self.filename = generate_contour_image()

        # Load image in grayscale
        self.img = cv2.imread(self.filename)

        self.lower_black = np.array([0, 0, 0])
        self.upper_black = np.array([50, 50, 50])
        self.mask = cv2.inRange(self.img, self.lower_black, self.upper_black)

        # Convert to boolean and skeletonize
        self.skeleton = skeletonize(self.mask > 0)
        self.skeleton_uint8 = (self.skeleton * 255).astype(np.uint8)

        # Find contours
        self.contours, _ = cv2.findContours(self.skeleton_uint8, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        self.half_contours = self.contours[::2]

        names = []
        self.count = 0

        # Image shape for normalization
        w,h = self.img.shape[1], self.img.shape[0]

        for contour in self.half_contours:
            pointset = PointSet2D(color=Color.RED)

            for pt in contour:
                x_pixel, y_pixel = pt[0]

                # Normalize pixel coords to [-1, 1]
                nx = 2 * (x_pixel / w) - 1
                ny = 2 * (y_pixel / h) - 1

                pointset.add(Point2D([nx, ny]))

            self.name = f"contour_{self.count}"
            self.addShape(pointset, self.name)
            names.append(self.name)
            self.count += 1

        self.names = names  # store for later if needed



        # def reset(self):
        #     # Generate terrain surface
        #     self.X, self.Y, self.Z = generate_contour_function()
        #     # Extract contour lines using matplotlib
        #     lv = np.linspace(self.Z.min(), self.Z.max(), 11)
        #     plt.contourf(self.X, self.Y, self.Z, levels=lv, cmap='coolwarm')  # fills the areas between contour lines
        #     self.cs = plt.contour(self.X, self.Y, self.Z, levels=lv, colors=['#000','#000']) # draws the contour lines black
            
        #     self.names = []
        #     self.count = 0

        #     for path in self.cs.get_paths():
        #         self.pointset = PointSet2D(color=Color.RED)
        #         for x, y in path.vertices:
        #             nx = 2 * x / 5 - 1
        #             ny = 2 * y / 5 - 1
        #             self.pointset.add(Point2D([nx, ny]))

        #         self.name = f"contour_{self.count}"
        #         self.addShape(self.pointset, self.name)  # shape first, name second
        #         self.names.append(self.name)                 # add to your list
                
        #         self.count += 1

        #     plt.close()


if __name__ == "__main__":
    app = Terrain()
    app.mainLoop()


