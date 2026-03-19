
import matplotlib.pyplot as plt
import numpy as np

def generate_contour_function():
    n = 100
    x = np.linspace(0, 5, n)
    y = np.linspace(0, 5, n)
    X, Y = np.meshgrid(x, y)


    # # Define peak center
    # x0, y0 = 2.5, 2.5

    # # Skewed/tilted Gaussian parameters
    # a = 1.0   # Width along x
    # b = 0.3   # Controls orientation
    # c = 0.5   # Width along y
    # # # Terrain-like function
    # Z = 2.0 * np.exp(-(a*(X - x0)**2 + 2*b*(X - x0)*(Y - y0) + c*(Y - y0)**2))
    Z = (
        1.8 * np.exp(-((X - 2.5)**2 / 0.5 + (Y - 3)**2 / 0.8)) +  # Main peak
        0.3 * np.exp(-((X - 3.5)**2 + (Y - 1.5)**2))             # Small bump to distort slope
    )
    # Z = 2.0 * np.exp(-((X - 2.5)**2 + (Y - 2.5)**2) / 0.5)
    # Z = (
    # 1.2 * np.exp(-((X - 1.5)**2 + (Y - 3.5)**2) / 0.3) +
    # 0.8 * np.exp(-((X - 3.5)**2 + (Y - 3)**2) / 0.4) +
    # 1.5 * np.exp(-((X - 2.5)**2 + (Y - 2)**2) / 0.6) +
    # 0.6 * np.exp(-((X - 1.2)**2 + (Y - 1.2)**2) )
    # )
    return X, Y ,Z

def generate_contour_image():
    X, Y, Z = generate_contour_function()
    plt.figure(figsize=(10, 10))
    lv = np.linspace(Z.min(), Z.max(), 7)
    plt.contourf(X, Y, Z, levels=lv, cmap='coolwarm')  # fills the areas between contour lines
    plt.contour(X, Y, Z, levels=lv, colors=['#000','#000']) # draws the contour lines black
    
    plt.axis('off')

    # Save  cropped image with transparent background
    filename = "contour_only.png"
    img = plt.savefig(filename, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.draw()
    plt.close()
    return filename
