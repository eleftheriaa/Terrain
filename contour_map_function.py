
import matplotlib.pyplot as plt
import numpy as np

def generate_contour_function():
    n = 100
    x = np.linspace(0, 5, n)
    y = np.linspace(0, 5, n)
    X, Y = np.meshgrid(x, y)

    # Terrain-like function
    Z = (
    1.2 * np.exp(-((X - 1.5)**2 + (Y - 3.5)**2) / 0.3) +
    0.8 * np.exp(-((X - 3.5)**2 + (Y - 3)**2) / 0.4) +
    1.5 * np.exp(-((X - 2.5)**2 + (Y - 2)**2) / 0.6) +
    0.6 * np.exp(-((X - 1.2)**2 + (Y - 1.2)**2) )
    )
    return X, Y ,Z

def generate_contour_image():
    X, Y, Z = generate_contour_function()
    plt.figure(figsize=(10, 10))
    lv = np.linspace(Z.min(), Z.max(), 10)
    plt.contourf(X, Y, Z, levels=lv, cmap='coolwarm') # color the contents
    plt.contour(X, Y, Z, levels=lv, colors=['#000','#000']) # color the lines
    
    plt.axis('off')

    # Save  cropped image with transparent background
    filename = "contour_only.png"
    img = plt.savefig(filename, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.draw()
    plt.close()
    return filename

# image = cv2.imread(filename)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print(f"Found {len(contours)} closed curves")

# cv2.drawContours(image, contours, -1, (0,0,255), 2)  # Red contours, 2px thick
# cv2.imshow('Contours', image)
# cv2.waitKey(0)  # Press any key to close

# # Colored Plot
# plt.figure(figsize=(6, 6))
# lv = np.linspace(np.min(Z), np.max(Z), 10)

# plt.contourf(X, Y, Z, levels=lv,cmap='Reds')
# plt.contour(X, Y, Z,levels=lv,colors=['#000','#000'])

# plt.show()
