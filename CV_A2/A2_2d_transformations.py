import cv2
import numpy as np

def get_transformed_image(img, matrix):
    rows=img.shape[0]
    cols=img.shape[1]

    x, y=np.meshgrid(np.arange(cols)-400, np.arange(rows)-400)
    coor=np.stack((x.flatten(), y.flatten(), np.ones(rows*cols)))
    new_coor=np.dot(matrix, coor).astype(int)

    new_coor[0, :]=new_coor[0, :]+400
    new_coor[1, :]=new_coor[1, :]+400

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


image=np.ones((801, 801), dtype=np.uint8)*255
# Please change the path of smile image if reading failed!
smile=cv2.imread('CV_Assignment_2_Images/smile.png',  cv2.IMREAD_GRAYSCALE)
image[350:451, 345:456]=smile
img=image.copy()
Img=img.copy()

while True:
  cv2.imshow("2D transformation", Img)
  key=cv2.waitKey(1)

  if key==ord("q"):
    break
  elif key==ord("h"):
    img=image.copy()
  elif key==ord("a"):
    matrix=np.array([[1, 0, 5],
                    [0, 1, 0],
                    [0, 0, 1]])
    img=get_transformed_image(img, matrix)
  elif key==ord("d"):
    matrix=np.array([[1, 0, -5],
                    [0, 1, 0],
                    [0, 0, 1]])
    img=get_transformed_image(img, matrix)
  elif key==ord("w"):
    matrix=np.array([[1, 0, 0],
                    [0, 1, 5],
                    [0, 0, 1]])
    img=get_transformed_image(img, matrix)
  elif key==ord("s"):
    matrix=np.array([[1, 0, 0],
                    [0, 1, -5],
                    [0, 0, 1]])
    img=get_transformed_image(img, matrix)
  elif key==ord("r"):
    matrix=np.array([[np.cos(np.deg2rad(5)), -np.sin(np.deg2rad(5)), 0],
                    [np.sin(np.deg2rad(5)), np.cos(np.deg2rad(5)), 0],
                    [0, 0, 1]])
    img=get_transformed_image(img, matrix)
  elif key==ord("t"):
    matrix=np.array([[np.cos(np.deg2rad(5)), np.sin(np.deg2rad(5)), 0],
                    [-np.sin(np.deg2rad(5)), np.cos(np.deg2rad(5)), 0],
                    [0, 0, 1]])
    img=get_transformed_image(img, matrix)
  elif key==ord("f"):
    matrix=np.array([[-1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])
    img=get_transformed_image(img, matrix)
  elif key==ord("g"):
    matrix=np.array([[1, 0, 0],
                    [0,-1, 0],
                    [0, 0, 1]])
    img=get_transformed_image(img, matrix)
  elif key==ord("x"):
    matrix=np.array([[1+0.05, 0, 0],
                [0, 1, 0],
                [0, 0, 1]])
    img=get_transformed_image(img, matrix)
  elif key==ord("c"):
    matrix=np.array([[1-0.05, 0, 0],
                [0, 1, 0],
                [0, 0, 1]])
    img=get_transformed_image(img, matrix)
  elif key==ord("y"):
    matrix=np.array([[1, 0, 0],
                    [0, 1+0.05, 0],
                    [0, 0, 1]])
    img=get_transformed_image(img, matrix)
  elif key==ord("u"):
    matrix=np.array([[1, 0, 0],
                    [0, 1-0.05, 0],
                    [0, 0, 1]])
    img=get_transformed_image(img, matrix)

  Img=img.copy()
  cv2.arrowedLine(Img, (0, 400), (800, 400), 0, 2)
  cv2.arrowedLine(Img, (400, 800), (400, 0), 0, 2)