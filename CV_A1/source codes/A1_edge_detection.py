import time
import numpy as np
import cv2
from A1_functions import get_gaussian_filter_2d
from A1_functions import cross_correlation_2d

def compute_image_gradient(img):
  sobel_x=np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])
  sobel_y=np.array([[1, 2, 1],
                    [ 0,  0,  0],
                    [-1, -2, -1]])

  img_dx=cross_correlation_2d(img, sobel_x)
  img_dy=cross_correlation_2d(img, sobel_y)

  mag=np.zeros_like(img)
  dir=np.zeros_like(img)
  for i in range(mag.shape[0]):
    for j in range(mag.shape[1]):
      mag[i, j]=np.sqrt(img_dx[i, j]**2+img_dy[i, j]**2)
      dir[i, j]=np.arctan2(img_dy[i, j],img_dx[i, j])

  return mag, dir

def non_maximum_suppression_dir ( mag , dir ):
  suppressed_mag=np.zeros((mag.shape[0], mag.shape[1]))

  ang=dir*180/np.pi
  ang[ang<0]+=180

  for i in range(1, mag.shape[0]-1):
    for j in range(1, mag.shape[1]-1):
      if (0<=ang[i, j]<22.5) or (157.5<=ang[i, j]<=180):
        if mag[i, j]>=mag[i, j-1] and mag[i, j]>=mag[i, j+1]:
          suppressed_mag[i, j]=mag[i, j]
      elif 22.5<=ang[i, j]<67.5:
        if mag[i, j]>=mag[i+1, j-1] and mag[i, j]>=mag[i-1, j+1]:
          suppressed_mag[i, j]=mag[i, j]
      elif 67.5<=ang[i, j]<112.5:
        if mag[i, j]>=mag[i+1, j] and mag[i, j]>=mag[i-1, j]:
          suppressed_mag[i, j]=mag[i, j]
      elif 112.5<=ang[i, j]<157.5:
        if mag[i, j]>=mag[i-1, j-1] and mag[i, j]>=mag[i+1, j+1]:
          suppressed_mag[i, j]=mag[i, j]

  return suppressed_mag

kernel=get_gaussian_filter_2d(7, 1.5)

shapes=cv2.imread("./A1_Images/shapes.png", cv2.IMREAD_GRAYSCALE)

g_shapes=cross_correlation_2d(shapes, kernel)

start=time.time()
shapes_mag, shapes_dir=compute_image_gradient(g_shapes)
end=time.time()
print("raw shapes computational time:", end-start)

cv2.imwrite("./result/part_2_edge_raw_shapes.png", shapes_mag)
shapes_MAG=cv2.imread("./result/part_2_edge_raw_shapes.png")
cv2.imshow("part_2_edge_raw_shapes.png", shapes_MAG)
cv2.waitKey(0)

lenna=cv2.imread("./A1_Images/lenna.png", cv2.IMREAD_GRAYSCALE)

g_lenna=cross_correlation_2d(lenna, kernel)

start=time.time()
lenna_mag, lenna_dir=compute_image_gradient(g_lenna)
end=time.time()
print("raw lenna computational time:", end-start)

cv2.imwrite("./result/part_2_edge_raw_lenna.png", lenna_mag)
lenna_MAG=cv2.imread("./result/part_2_edge_raw_lenna.png")
cv2.imshow("part_2_edge_raw_lenna.png", lenna_MAG)
cv2.waitKey(0)

start=time.time()
suppressed_shapes=non_maximum_suppression_dir(shapes_mag, shapes_dir)
end=time.time()
print("suppressed shapes computational time:", end-start)

cv2.imwrite("./result/part_2_edge_sup_shapes.png", suppressed_shapes)
shapes_SUP=cv2.imread("./result/part_2_edge_sup_shapes.png")
cv2.imshow("part_2_edge_sup_shapes.png", shapes_SUP)
cv2.waitKey(0)

start=time.time()
suppressed_lenna=non_maximum_suppression_dir(lenna_mag, lenna_dir)
end=time.time()
print("suppressed lenna computational time:", end-start)

cv2.imwrite("./result/part_2_edge_sup_lenna.png", suppressed_lenna)
lenna_SUP=cv2.imread("./result/part_2_edge_sup_lenna.png")
cv2.imshow("part_2_edge_sup_lenna.png", lenna_SUP)
cv2.waitKey(0)