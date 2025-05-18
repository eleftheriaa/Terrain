# import cv2
# from contour_map_function import  generate_contour_image

# # image = cv2.imread("contour_maps/blurred2.png")
# filename = generate_contour_image()

# image = cv2.imread(filename)
# for i,im in enumerate(image):
#     print(f"row{i}:",im)


import cv2
from vvrpywork.constants import Key, Mouse, Color
from vvrpywork.scene import Scene2D, Scene3D, get_rotation_matrix, world_space
from vvrpywork.shapes import (
    Point2D, Line2D, Triangle2D, Circle2D, Rectangle2D,
    PointSet2D, LineSet2D, Polygon2D, Label2D,
    Point3D, Line3D, Arrow3D, Sphere3D, Cuboid3D, Cuboid3DGeneralized,
    PointSet3D, LineSet3D, Mesh3D, Label3D
)

from contour_map_function import  generate_contour_image

WIDTH = 739
HEIGHT = 744

class Terrain(Scene2D):
    def __init__(self):
        super().__init__(WIDTH, HEIGHT, "Terrain")
        self.points:list[Point2D] = []

        self.reset() # scene initialiazation

    def reset(self):
        filename = generate_contour_image()
        image = cv2.imread(filename)   # BGR format!
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert to RGB

        height, width, _ = image.shape
        print(height,width)

        # Option 1: Group by color (for unique PointSet2D per color)
        color_to_points = {}

        for y in range(HEIGHT):
            for x in range(WIDTH):
                color = tuple(image[y, x])  # RGB tuple
                if color not in color_to_points:
                    color_to_points[color] = []
                color_to_points[color].append((x, height - y-1))  # flip Y to match scene coords

        # Now create a PointSet2D for each color
        point_sets = []
        for color, pts in color_to_points.items():
            ps = PointSet2D(points=pts, size=1, color=tuple(c/255 for c in color))  # normalize to [0, 1]
            point_sets.append(ps)
        for ps in point_sets:
            self.addShape(ps)
    
if __name__ == "__main__":
    app = Terrain()
    app.mainLoop()


