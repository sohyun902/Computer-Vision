import cv2
import numpy as np
import time

def cross_correlation_1d(img, kernel):
  if kernel.shape[0]==1: # if kernel is horizontal
    # padding
    pad_w=int((kernel.shape[1]-1)/2)
    pad_img=np.zeros((img.shape[0], img.shape[1]+2*pad_w))
    pad_img[:, pad_w:-pad_w]=img

    pad_img[:, :pad_w]=np.tile(img[:, 0], (pad_w, 1)).transpose()
    pad_img[:, -pad_w:]=np.tile(img[:, -1], (pad_w, 1)).transpose()

    k_size=kernel.shape[1]
    filtered_img=np.zeros((img.shape[0], img.shape[1]))
    for i in range(pad_img.shape[0]):
      for j in range(pad_img.shape[1]-k_size+1):
        filtered_img[i, j]=(pad_img[i, j:j+k_size]*kernel).sum()

  elif kernel.shape[1]==1:  # if kernel is vertical
    # padding
    pad_w=int((kernel.shape[0]-1)/2)
    pad_img=np.zeros((img.shape[0]+2*pad_w, img.shape[1]))
    pad_img[pad_w:-pad_w, :]=img

    pad_img[:pad_w, :]=np.tile(img[0, :], (pad_w, 1))
    pad_img[-pad_w:, :]=np.tile(img[-1, :], (pad_w, 1))

    k_size=kernel.shape[0]
    filtered_img=np.zeros((img.shape[0], img.shape[1]))
    for i in range(pad_img.shape[0]-k_size+1):
      for j in range(pad_img.shape[1]):
       filtered_img[i, j]=(pad_img[i:i+k_size, j].reshape(-1, 1)*kernel).sum()

  return filtered_img

def cross_correlation_2d(img, kernel):
  # padding
  pad_w=int((kernel.shape[0]-1)/2)
  pad_img=np.zeros((img.shape[0]+2*pad_w, img.shape[1]+2*pad_w))
  pad_img[pad_w:-pad_w, pad_w:-pad_w]=img
  pad_img[:pad_w, :pad_w]=img[0, 0]
  pad_img[:pad_w, -pad_w:]=img[0, -1]
  pad_img[-pad_w:, :pad_w]=img[-1, 0]
  pad_img[-pad_w: ,-pad_w:]=img[-1, -1]

  top=np.tile(img[0, :], (pad_w, 1))
  bottom=np.tile(img[-1, :], (pad_w, 1))
  left=np.tile(img[:, 0], (pad_w, 1)).transpose()
  right=np.tile(img[:, -1], (pad_w, 1)).transpose()

  pad_img[:pad_w, pad_w:-pad_w]=top
  pad_img[-pad_w:, pad_w:-pad_w]=bottom
  pad_img[pad_w:-pad_w, :pad_w]=left
  pad_img[pad_w:-pad_w, -pad_w:]=right

  k_size=kernel.shape[0]
  filtered_img=np.zeros((img.shape[0], img.shape[1]))
  for i in range(pad_img.shape[0]-k_size+1):
    for j in range(pad_img.shape[1]-k_size+1):
      filtered_img[i, j]=(pad_img[i:i+k_size, j:j+k_size]*kernel).sum()
  return filtered_img

def get_gaussian_filter_1d(size, sigma):
  dist=np.arange((size//2)*(-1), (size//2)+1)
  arr=np.zeros((size))
  for i in range(size):
    arr[i]=dist[i]
    
  kernel=np.exp(-(arr**2)/(2*sigma**2))
  kernel=kernel/kernel.sum()
  kernel=np.array([kernel])
  return kernel

def get_gaussian_filter_2d(size, sigma):
  dist=np.arange((size//2)*(-1), (size//2)+1)
  arr=np.zeros((size, size))
  for i in range(size):
    for j in range(size):
        arr[i, j]=dist[i]**2+dist[j]**2

  kernel=np.zeros((size, size))
  for i in range(size):
    for j in range(size):
      kernel[i, j]=np.exp(-(arr[i, j]/(2*sigma**2)))
  kernel=kernel/kernel.sum()
  return kernel

print("gaussian filter 1d:", get_gaussian_filter_1d(5, 1))
print("gaussian filter 2d:", get_gaussian_filter_2d(5, 1))

kernel1=get_gaussian_filter_2d(5, 1)
kernel2=get_gaussian_filter_2d(5, 6)
kernel3=get_gaussian_filter_2d(5, 11)
kernel4=get_gaussian_filter_2d(11, 1)
kernel5=get_gaussian_filter_2d(11, 6)
kernel6=get_gaussian_filter_2d(11, 11)
kernel7=get_gaussian_filter_2d(17, 1)
kernel8=get_gaussian_filter_2d(17, 6)
kernel9=get_gaussian_filter_2d(17, 11)

shapes=cv2.imread("./A1_Images/shapes.png", cv2.IMREAD_GRAYSCALE)

shapes_1=cv2.putText(cross_correlation_2d(shapes, kernel1), "5x5 s=1", (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2, cv2.LINE_AA)
shapes_2=cv2.putText(cross_correlation_2d(shapes, kernel2), "5x5 s=6", (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2, cv2.LINE_AA)
shapes_3=cv2.putText(cross_correlation_2d(shapes, kernel3), "5x5 s=11", (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2, cv2.LINE_AA)
shapes_4=cv2.putText(cross_correlation_2d(shapes, kernel4), "11x11 s=1", (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2, cv2.LINE_AA)
shapes_5=cv2.putText(cross_correlation_2d(shapes, kernel5), "11x11 s=6", (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2, cv2.LINE_AA)
shapes_6=cv2.putText(cross_correlation_2d(shapes, kernel6), "11x11 s=11", (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2, cv2.LINE_AA)
shapes_7=cv2.putText(cross_correlation_2d(shapes, kernel7), "17x17 s=1", (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2, cv2.LINE_AA)
shapes_8=cv2.putText(cross_correlation_2d(shapes, kernel8), "17x17 s=6", (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2, cv2.LINE_AA)
shapes_9=cv2.putText(cross_correlation_2d(shapes, kernel9), "17x17 s=11", (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2, cv2.LINE_AA)

shapes_first=cv2.hconcat([shapes_1, shapes_2, shapes_3])
shapes_second=cv2.hconcat([shapes_4, shapes_5, shapes_6])
shapes_third=cv2.hconcat([shapes_7, shapes_8, shapes_9]) 

shapes_concat=cv2.vconcat([shapes_first, shapes_second, shapes_third])
cv2.imwrite("./result/part_1_gaussian_filtered_shapes.png", shapes_concat)
Shapes_concat=cv2.imread("./result/part_1_gaussian_filtered_shapes.png")
Shapes_concat=cv2.resize(Shapes_concat, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
cv2.imshow("part_1_gaussian_filtered_shapes.png", Shapes_concat)
cv2.waitKey(0)

lenna=cv2.imread("./A1_Images/lenna.png", cv2.IMREAD_GRAYSCALE)

lenna_1=cv2.putText(cross_correlation_2d(lenna, kernel1), "5x5 s=1", (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2, cv2.LINE_AA)
lenna_2=cv2.putText(cross_correlation_2d(lenna, kernel2), "5x5 s=6", (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2, cv2.LINE_AA)
lenna_3=cv2.putText(cross_correlation_2d(lenna, kernel3), "5x5 s=11", (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2, cv2.LINE_AA)
lenna_4=cv2.putText(cross_correlation_2d(lenna, kernel4), "11x11 s=1", (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2, cv2.LINE_AA)
lenna_5=cv2.putText(cross_correlation_2d(lenna, kernel5), "11x11 s=6", (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2, cv2.LINE_AA)
lenna_6=cv2.putText(cross_correlation_2d(lenna, kernel6), "11x11 s=11", (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2, cv2.LINE_AA)
lenna_7=cv2.putText(cross_correlation_2d(lenna, kernel7), "17x17 s=1", (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2, cv2.LINE_AA)
lenna_8=cv2.putText(cross_correlation_2d(lenna, kernel8), "17x17 s=6", (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2, cv2.LINE_AA)
lenna_9=cv2.putText(cross_correlation_2d(lenna, kernel9), "17x17 s=11", (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2, cv2.LINE_AA)

lenna_first=cv2.hconcat([lenna_1, lenna_2, lenna_3])
lenna_second=cv2.hconcat([lenna_4, lenna_5, lenna_6])
lenna_third=cv2.hconcat([lenna_7, lenna_8, lenna_9]) 

lenna_concat=cv2.vconcat([lenna_first, lenna_second, lenna_third])
cv2.imwrite("./result/part_1_gaussian_filtered_lenna.png", lenna_concat)
Lenna_concat=cv2.imread("./result/part_1_gaussian_filtered_lenna.png")
Lenna_concat=cv2.resize(Lenna_concat, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
cv2.imshow("part_1_gaussian_filtered_lenna.png", Lenna_concat)
cv2.waitKey(0)

h_kernel_1d=get_gaussian_filter_1d(17, 6)
v_kernel_1d=np.transpose(h_kernel_1d)
kernel_2d=get_gaussian_filter_2d(17, 6)

start=time.time()
shapes_h=cross_correlation_1d(shapes, v_kernel_1d)
shapes_h_v=cross_correlation_1d(shapes_h, h_kernel_1d)
end=time.time()
print("shapes 1D computational time:", end-start)

start=time.time()
shapes_2d=cross_correlation_2d(shapes, kernel_2d)
end=time.time()
print("shapes 2D computational time:", end-start)

s_diff_map=np.abs(shapes_2d-shapes_h_v)
cv2.imshow("part_1_gaussian_filtered_shapes.png", s_diff_map)
cv2.waitKey(0)
print("sum of absolute intensity difference:", np.sum(s_diff_map))

start=time.time()
lenna_h=cross_correlation_1d(lenna, h_kernel_1d)
lenna_h_v=cross_correlation_1d(lenna_h, v_kernel_1d)
end=time.time()
print("lenna 1D computational time:", end-start)

start=time.time()
lenna_2d=cross_correlation_2d(lenna, kernel_2d)
end=time.time()
print("lenna 2D computational time:", end-start)

l_diff_map=np.abs(lenna_2d-lenna_h_v)
cv2.imshow("part_1_gaussian_filtered_lenna.png", l_diff_map)
cv2.waitKey(0)
print("sum of absolute intensity difference:", np.sum(l_diff_map))
