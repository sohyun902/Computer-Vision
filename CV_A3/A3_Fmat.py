
import cv2
import numpy as np
import random
import time
from CV_Assignment_3_Data.compute_avg_reproj_error import compute_avg_reproj_error

def normalize(points):
  global img
  center=(img.shape[1]/2, img.shape[0]/2)
  points_t=points-center
  Mean=np.array([[1, 0, -center[0]], 
                 [0, 1, -center[1]], 
                 [0, 0, 1]])

  x_max=center[0]
  y_max=center[1]
  x_min=-center[0]
  y_min=-center[1]

  points_s=2*(points_t-[x_min, y_min])/([x_max-x_min, y_max-y_min])-1

  one=2/(x_max-x_min)
  two=-1-one*x_min
  three=2/(y_max-y_min)
  four=-1-three*y_min

  Scale=np.array([[one, 0, two], 
                  [0, three, four], 
                  [0, 0, 1]])

  T=np.dot(Scale, Mean)

  return points_s, T

def compute_F_raw(M):
  left=M[:, 0:2]
  right=M[:, 2:4]

  A=[]
  for l, r in zip(left, right):
    a=[l[0]*r[0], r[0]*l[1], r[0], l[0]*r[1], l[1]*r[1], r[1], l[0], l[1], 1]
    A.append(a)

  A=np.array(A)
  u, s, v=np.linalg.svd(A)
  F=v[-1].reshape(3,3)

  return F

def compute_F_norm(M):
  left=M[:, 0:2]
  right=M[:, 2:4]

  norm_left, T1=normalize(left)
  norm_right, T2=normalize(right)

  A=[]
  for l, r in zip(norm_left, norm_right):
    a=[l[0]*r[0], r[0]*l[1], r[0], l[0]*r[1], l[1]*r[1], r[1], l[0], l[1], 1]
    A.append(a)

  A=np.array(A)
  u, s, v=np.linalg.svd(A)
  F=v[-1].reshape(3,3)

  U, S, V=np.linalg.svd(F)
  S[-1]=0
  F=np.dot(U, np.dot(np.diag(S), V))

  unnormalized_F=np.dot(T2.transpose(), np.dot(F, T1))
  return unnormalized_F

def compute_F_mine(M):

  np.random.seed(0)
  maxinliners=[]
  
  for i in range(3000):
    idx=np.random.choice(M.shape[0], size=8, replace=False)
    M_sample=M[idx]

    f=compute_F_norm(M_sample)
    inliners=[]

    for j in range(M.shape[0]):
      left=M[j][0:2]
      right=M[j][2:4]

      x=np.array([left[0], left[1], 1])
      x_prime=np.array([right[0], right[1], 1])

      error=np.dot(np.dot(x_prime, f), x)
      error=np.abs(error)

      if error<0.001:
        inliners.append([left[0], left[1], right[0], right[1]])

    if len(inliners)>len(maxinliners):
      maxinliners=inliners

  maxinliners=np.array(maxinliners)
  F=compute_F_norm(maxinliners)
  return F

def two_end_points(line, img):

  left=0
  right=img.shape[1]

  pt1_x=left
  pt1_y=-(line[2][0]/line[1][0])

  pt2_x=right
  pt2_y=(-line[0][0]*right-line[2][0])/line[1][0]

  pt1=(int(pt1_x), int(pt1_y))
  pt2=(int(pt2_x), int(pt2_y))

  return pt1, pt2

def visualize(M, F, img1, img2):

  red=(255, 0, 0)
  green=(0, 255, 0)
  blue=(0, 0, 255)

  colors=[red, green, blue]

  while True:
    idx=np.arange(M.shape[0])
    idx=list(idx)
    rand_idx=random.sample(idx, 3)

    rand_pts=M[rand_idx]

    left=rand_pts[:, 0:2]
    right=rand_pts[:, 2:4]

    p1=(left[0][0], left[0][1])
    p2=(left[1][0], left[1][1])
    p3=(left[2][0], left[2][1])
    q1=(right[0][0], right[0][1])
    q2=(right[1][0], right[1][1])
    q3=(right[2][0], right[2][1])

    l1=np.dot(F.transpose(), np.array([[q1[0]], [q1[1]], [1]]))
    l2=np.dot(F.transpose(), np.array([[q2[0]], [q2[1]], [1]]))
    l3=np.dot(F.transpose(), np.array([[q3[0]], [q3[1]], [1]]))
    m1=np.dot(F, np.array([[p1[0]], [p1[1]], [1]]))
    m2=np.dot(F, np.array([[p2[0]], [p2[1]], [1]]))
    m3=np.dot(F, np.array([[p3[0]], [p3[1]], [1]]))

    ps=[tuple(map(int, p1)), tuple(map(int, p2)), tuple(map(int, p3))]
    qs=[tuple(map(int, q1)), tuple(map(int, q2)), tuple(map(int, q3))]
    ls=[l1, l2, l3]
    ms=[m1, m2, m3]

    Img1=img1.copy()
    Img2=img2.copy()

    for i in range(len(colors)):
      pt1, pt2=two_end_points(ls[i], Img1)
      cv2.line(Img1, pt1, pt2, colors[i], 2)
      cv2.circle(Img1, ps[i], 3, colors[i], 2)
      pt1, pt2=two_end_points(ms[i], Img2)
      cv2.line(Img2, pt1, pt2, colors[i], 2)
      cv2.circle(Img2, qs[i], 3, colors[i], 2)
    
    Result=np.hstack((Img1, Img2))
    cv2.imshow("Epipolar lines", Result)
    key=cv2.waitKey(0)
    if key==ord('q'):
      break
    else:
      continue

# It is recommended to modify image or text paths according to your environment.

temple1=cv2.imread("./CV_Assignment_3_Data/temple1.png")
temple2=cv2.imread("./CV_Assignment_3_Data/temple2.png")

library1=cv2.imread("./CV_Assignment_3_Data/library1.jpg")
library2=cv2.imread("./CV_Assignment_3_Data/library2.jpg")

house1=cv2.imread("./CV_Assignment_3_Data/house1.jpg")
house2=cv2.imread("./CV_Assignment_3_Data/house2.jpg")

M_temple=np.loadtxt("./CV_Assignment_3_Data/temple_matches.txt")
img=temple1.copy()

F_temple_raw=compute_F_raw(M_temple)
F_temple_norm=compute_F_norm(M_temple)
start=time.time()
F_temple_mine=compute_F_mine(M_temple)
end=time.time()

print("Average Reprojection Errors (temple1.png and temple2.png)")
print(f"  Raw: {compute_avg_reproj_error(M_temple, F_temple_raw)}")
print(f"  Norm: {compute_avg_reproj_error(M_temple, F_temple_norm)}")
print(f"  Mine: {compute_avg_reproj_error(M_temple, F_temple_mine)}")

print()
print("Computational time:", end-start)
print()

M_library=np.loadtxt("./CV_Assignment_3_Data/library_matches.txt")
img=library1.copy()

F_library_raw=compute_F_raw(M_library)
F_library_norm=compute_F_norm(M_library)
start=time.time()
F_library_mine=compute_F_mine(M_library)
end=time.time()

print("Average Reprojection Errors (library1.png and library2.png)")
print(f"  Raw: {compute_avg_reproj_error(M_library, F_library_raw)}")
print(f"  Norm: {compute_avg_reproj_error(M_library, F_library_norm)}")
print(f"  Mine: {compute_avg_reproj_error(M_library, F_library_mine)}")

print()
print("Computational time:", end-start)
print()

M_house=np.loadtxt("./CV_Assignment_3_Data/house_matches.txt")
img=house1.copy()

F_house_raw=compute_F_raw(M_house)
F_house_norm=compute_F_norm(M_house)
start=time.time()
F_house_mine=compute_F_mine(M_house)
end=time.time()

print("Average Reprojection Errors (house1.jpg and house2.jpg)")
print(f"  Raw: {compute_avg_reproj_error(M_house, F_house_raw)}")
print(f"  Norm: {compute_avg_reproj_error(M_house, F_house_norm)}")
print(f"  Mine: {compute_avg_reproj_error(M_house, F_house_mine)}")

print()
print("Computational time:", end-start)
print()

visualize(M_temple, F_temple_mine, temple1, temple2)
visualize(M_library, F_library_mine, library1, library2)
visualize(M_house, F_house_mine, house1, house2)