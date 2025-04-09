import numpy as np

def get_transformed_image(img, matrix):
    rows=img.shape[0]
    cols=img.shape[1]

    x, y=np.meshgrid(np.arange(cols), np.arange(rows))
    coor=np.stack((x.flatten(), y.flatten(), np.ones(rows*cols)))

    new_coor=np.dot(matrix, coor).astype(int)

    new_coor[0, :]=new_coor[0, :]
    new_coor[1, :]=new_coor[1, :]

    for i in range(len(new_coor[0])):
      if new_coor[0][i]<0:
        new_coor[0][i]=0
      elif new_coor[0][i]>cols-1:
        new_coor[0][i]=cols-1

    for j in range(len(new_coor[1])):
      if new_coor[1][j]<0:
        new_coor[1][j]=0
      elif new_coor[1][j]>rows-1:
        new_coor[1][j]=rows-1

    transformed_img=img[new_coor[1, :], new_coor[0, :]]
    transformed_img=transformed_img.reshape(img.shape)

    return transformed_img

def resize(img1, img2):
    matrix=np.array([[img1.shape[1]/img2.shape[1], 0, 0],
                [0, img1.shape[0]/img2.shape[0], 0],
                [0, 0, 1]])
  
    src=np.ones((801, 801), dtype=np.uint8)*255
    src[:img1.shape[0], :img1.shape[1]]=img1

    dest=np.ones((801, 801), dtype=np.uint8)*255
    dest[:img2.shape[0], :img2.shape[1]]=img2

    new=get_transformed_image(src,  matrix)
    return new[:img2.shape[0], :img2.shape[1]]