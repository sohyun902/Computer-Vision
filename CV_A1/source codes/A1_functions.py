import numpy as np

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