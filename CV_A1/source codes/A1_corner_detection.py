import numpy as np
import cv2
import time
from A1_functions import get_gaussian_filter_2d
from A1_functions import cross_correlation_2d

def compute_corner_response(img):
  sobel_x=np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])
  sobel_y=np.array([[1, 2, 1],
                    [ 0,  0,  0],
                    [ -1,  -2, -1]])

  dx=cross_correlation_2d(img, sobel_x)
  dy=cross_correlation_2d(img, sobel_y)

  Ixx=cross_correlation_2d(dx*dx, np.ones((5, 5)))
  Ixy=cross_correlation_2d(dx*dy, np.ones((5, 5)))
  Iyy=cross_correlation_2d(dy*dy, np.ones((5, 5)))

  k=float(0.04)
  R=(Ixx*Iyy-Ixy**2)-k*(Ixx+Iyy)**2
  R[R<0]=0
  R=R/np.max(R)

  return R

def non_maximum_suppression_win(R, winSize=11):
  threshold=0.1
  pad_w = (winSize - 1) // 2
  pad_R=np.zeros((R.shape[0]+2*pad_w, R.shape[1]+2*pad_w))
  pad_R[pad_w:-pad_w, pad_w:-pad_w]=R
  suppressed_R = np.zeros_like(R)

  for i in range(pad_w, pad_R.shape[0] - pad_w):
    for j in range(pad_w, pad_R.shape[1] - pad_w):
      if pad_R[i, j]==np.max(pad_R[i-pad_w:i+pad_w+1, j-pad_w:j+pad_w+1]) and pad_R[i, j]>threshold:
        suppressed_R[i-pad_w, j-pad_w] = 255

  return suppressed_R


kernel=get_gaussian_filter_2d(7, 1.5)

shapes=cv2.imread("./A1_Images/shapes.png", cv2.IMREAD_GRAYSCALE)

g_shapes=cross_correlation_2d(shapes, kernel)

start=time.time()
shapes_R=compute_corner_response(g_shapes)
end=time.time()
print("raw shapes computational time:", end-start)

shapes_r=shapes_R*255
cv2.imwrite("./result/part_3_corner_raw_shapes.png", shapes_r)
shapes_RR=cv2.imread("./result/part_3_corner_raw_shapes.png")
cv2.imshow("part_3_corner_raw_shapes.png", shapes_RR)
cv2.waitKey(0)

lenna=cv2.imread("./A1_Images/lenna.png", cv2.IMREAD_GRAYSCALE)

g_lenna=cross_correlation_2d(lenna, kernel)

start=time.time()
lenna_R=compute_corner_response(g_lenna)
end=time.time()
print("raw lenna computational time:", end-start)

lenna_r=lenna_R*255
cv2.imwrite("./result/part_3_corner_raw_lenna.png", lenna_r)
lenna_RR=cv2.imread("./result/part_3_corner_raw_lenna.png")
cv2.imshow("part_3_corner_raw_lenna.png", lenna_RR)
cv2.waitKey(0)

shapes_corners=np.zeros_like(shapes)
shapes_corners[shapes_R>0.1]=255
x, y=np.where(shapes_corners==255)
shapes_bgr=cv2.cvtColor(shapes.copy(), cv2.COLOR_GRAY2BGR)
shapes_bgr[x, y]=(0, 255, 0)

cv2.imwrite("./result/part_3_corner_bin_shapes.png", shapes_bgr)
shapes_BGR=cv2.imread("./result/part_3_corner_bin_shapes.png")
cv2.imshow("part_3_corner_bin_shapes.png", shapes_BGR)
cv2.waitKey(0)

lenna_corners=np.zeros_like(lenna)
lenna_corners[lenna_R>0.1]=255
x, y=np.where(lenna_corners==255)
lenna_bgr=cv2.cvtColor(lenna.copy(), cv2.COLOR_GRAY2BGR)
lenna_bgr[x, y]=(0, 255, 0)

cv2.imwrite("./result/part_3_corner_bin_lenna.png", lenna_bgr)
lenna_BGR=cv2.imread("./result/part_3_corner_bin_lenna.png")
cv2.imshow("part_3_corner_bin_lenna.png", lenna_BGR)
cv2.waitKey(0)

start=time.time()
shapes_sup_R=non_maximum_suppression_win(shapes_R)
end=time.time()
print("suppressed shapes computational time:", end-start)

corners=np.where(shapes_sup_R==255)
shapes_bgr=cv2.cvtColor(shapes.copy(), cv2.COLOR_GRAY2BGR)
for i, j in zip(corners[1], corners[0]):
   cv2.circle(shapes_bgr, (i, j), 5, (0, 255, 0), 2)

cv2.imwrite("./result/part_3_corner_sup_shapes.png", shapes_bgr)
shapes_SBGR=cv2.imread("./result/part_3_corner_sup_shapes.png")
cv2.imshow("part_3_corner_sup_shapes.png", shapes_SBGR)
cv2.waitKey(0)

start=time.time()
lenna_sup_R=non_maximum_suppression_win(lenna_R)
end=time.time()
print("suppressed lenna computational time:", end-start)

corners=np.where(lenna_sup_R==255)
lenna_bgr=cv2.cvtColor(lenna.copy(), cv2.COLOR_GRAY2BGR)
for i, j in zip(corners[1], corners[0]):
   cv2.circle(lenna_bgr, (i, j), 5, (0, 255, 0), 2)

cv2.imwrite("./result/part_3_corner_sup_lenna.png", lenna_bgr)
lenna_SBGR=cv2.imread("./result/part_3_corner_sup_lenna.png")
cv2.imshow("part_3_corner_sup_lenna.png", lenna_SBGR)
cv2.waitKey(0)