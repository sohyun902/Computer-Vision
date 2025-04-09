import cv2
import numpy as np
import random
random.seed(0)
import time
from A2_function import resize

def match(img1, img2):
  orb=cv2.ORB_create()

  kp1=orb.detect(img1, None)
  kp1, des1 = orb.compute(img1, kp1)

  kp2=orb.detect(img2, None)
  kp2, des2=orb.compute(img2, kp2)

  matches = []
  for i in range(len(des2)):
      min_dist=float('inf')
      min_idx=-1
      for j in range(len(des1)):
          dist=np.sum(np.bitwise_xor(des2[i], des1[j]))
          dist=float(dist)
          if dist < min_dist:
              min_dist=dist
              min_idx=j
      matches.append(cv2.DMatch(i, min_idx, min_dist))

  matches=sorted(matches, key=lambda x: x.distance)
  return matches, kp1, kp2

def points(matches):
  srcp=[]
  destp=[]

  for m in matches:
    Xs, Ys=kp1[m.trainIdx].pt  
    Xd, Yd=kp2[m.queryIdx].pt  
    srcp.append([Xs, Ys])
    destp.append([Xd, Yd])

  srcp=np.array(srcp)
  destp=np.array(destp)

  src_mean=np.mean(srcp, axis=0)
  src_sub_mean=srcp-src_mean
  src_max_distance=0
  for i in range(len(src_sub_mean)):
    dist=np.sqrt(np.sum(src_sub_mean[i]**2))
    if dist>src_max_distance:
      src_max_distance=dist
  src_scale=np.sqrt(2)/src_max_distance
  scaled_srcp=src_sub_mean*src_scale

  dest_mean=np.mean(destp, axis=0)
  dest_sub_mean=destp-dest_mean
  dest_max_distance=0
  for i in range(len(dest_sub_mean)):
    dist=np.sqrt(np.sum(dest_sub_mean[i]**2))
    if dist>dest_max_distance:
      dest_max_distance=dist
  dest_scale=np.sqrt(2)/dest_max_distance
  scaled_destp=dest_sub_mean*dest_scale 

  Ts=np.array([[src_scale, 0, -src_mean[0]*src_scale],
                    [0, src_scale, -src_mean[1]*src_scale],
                    [0, 0, 1]])
  Td=np.array([[dest_scale, 0, -dest_mean[0]*dest_scale],
                    [0, dest_scale, -dest_mean[1]*dest_scale],
                    [0, 0, 1]])
  
  return scaled_srcp[:32], scaled_destp[:32], Ts, Td

def compute_homography(srcP, destP):
  A=[]
  for s, d in zip(srcP, destP):
    p1=np.array([s[0], s[1], 1])
    p2=np.array([d[0], d[1], 1])

    a1=[-p1[0], -p1[1], -1, 0, 0, 0, p1[0]*p2[0], p1[1]*p2[0], p2[0]]
    a2=[0, 0, 0, -p1[0], -p1[1], -1, p1[0]*p2[1], p1[1]*p2[1], p2[1]]

    A.append(a1)
    A.append(a2)

  A=np.array(A)

  u, s, v=np.linalg.svd(A)
  H=v[-1]
  H=H.reshape(3, 3)

  return H

def compute_homography_ransac(srcP, destP, th):

  maxinliners=[]

  def dist(s, d, h):
    p1=np.transpose(np.array([[s[0], s[1], 1]]))
    estimated_p2=np.dot(h, p1)
    estimated_p2=estimated_p2/estimated_p2[1][0]

    p2=np.transpose(np.array([[d[0], d[1], 1]]))
    error=np.sqrt(np.sum((p2-estimated_p2)**2))

    return error

  for i in range(7000):
    srcp=[]
    destp=[]

    one=random.randrange(0, len(srcP))
    two=random.randrange(0, len(srcP))
    three=random.randrange(0, len(srcP))
    four=random.randrange(0, len(srcP))

    srcp.append(srcP[one])
    srcp.append(srcP[two])
    srcp.append(srcP[three])
    srcp.append(srcP[four])

    destp.append(srcP[one])
    destp.append(srcP[two])
    destp.append(srcP[three])
    destp.append(srcP[four])

    srcp=np.array(srcp)
    destp=np.array(destp)

    h=compute_homography(srcp, destp)
    inliners=[]

    for j in range(len(srcP)):
      d=dist(srcP[j], destP[j], h)
      if d<th:
        inliners.append([srcP[j], destP[j]])

    if len(inliners)>len(maxinliners):
      maxinliners=inliners

  maxinliners=np.array(maxinliners)
  in_s=[]
  in_d=[]
  for i in range(len(maxinliners)):
    in_s.append(maxinliners[i][0])
    in_d.append(maxinliners[i][1])
  in_s=np.array(in_s)
  in_d=np.array(in_d)

  H=compute_homography(in_s, in_d)

  return H

# Please change paths of images if reading failed!
cover=cv2.imread('CV_Assignment_2_Images/cv_cover.jpg', cv2.IMREAD_GRAYSCALE)
desk=cv2.imread('CV_Assignment_2_Images/cv_desk.png', cv2.IMREAD_GRAYSCALE)
hp_cover=cv2.imread('CV_Assignment_2_Images/hp_cover.jpg', cv2.IMREAD_GRAYSCALE)
d10=cv2.imread('CV_Assignment_2_Images/diamondhead-10.png', cv2.IMREAD_GRAYSCALE)
d11=cv2.imread('CV_Assignment_2_Images/diamondhead-11.png', cv2.IMREAD_GRAYSCALE)

orb=cv2.ORB_create()

matches, kp1, kp2=match(cover, desk)

matched_img=cv2.drawMatches(desk, kp2, cover, kp1, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow("feature matching", matched_img)
cv2.waitKey(0)

scaled_srcp, scaled_destp, Ts, Td=points(matches)

HN=compute_homography(scaled_srcp, scaled_destp)
warped_img=cv2.warpPerspective(cover, np.dot(np.dot(np.linalg.inv(Td), HN), Ts), (desk.shape[1], desk.shape[0]))
cv2.imshow("Homography with normalization", warped_img)
cv2.waitKey(0)
#cv2.imwrite("./result/A2_Homography with normalization.png", warped_img)

wrapped_HN=desk.copy()

for i in range(desk.shape[0]):
  for j in range(desk.shape[1]):
    if warped_img[i][j]!=0:
      wrapped_HN[i][j]=warped_img[i][j]
cv2.imshow("Homography with normalization", wrapped_HN)
cv2.waitKey(0)
#cv2.imwrite("./result/A2_Homography with normalization_wrapped.png", wrapped_HN)

t=time.time()
HR=compute_homography_ransac(scaled_srcp, scaled_destp, 1.1)
print("computing time of homography with RANSAC:",time.time()-t)
warped_img=cv2.warpPerspective(cover, np.dot(np.dot(np.linalg.inv(Td), HR), Ts), (desk.shape[1], desk.shape[0]))
cv2.imshow("Homography with RANSAC", warped_img)
cv2.waitKey(0)
#cv2.imwrite("./result/A2_Homography with RANSAC.png", warped_img)

wrapped_HR=desk.copy()

for i in range(desk.shape[0]):
  for j in range(desk.shape[1]):
    if warped_img[i][j]!=0:
      wrapped_HR[i][j]=warped_img[i][j]
cv2.imshow("Homography with RANSAC", wrapped_HR)
cv2.waitKey(0)
#cv2.imwrite("./result/A2_Homography with RANSAC_wrapped.png", wrapped_HR)

hp_cover=resize(hp_cover, cover)

warped_hp = cv2.warpPerspective(hp_cover, np.dot(np.dot(np.linalg.inv(Td), HR), Ts), (desk.shape[1], desk.shape[0]))
cv2.imshow("Homography with RANSAC", warped_hp)
cv2.waitKey(0)
#cv2.imwrite("./result/A2_hp_Homography with RANSAC.png", warped_hp)

wrapped_HR=desk.copy()

for i in range(desk.shape[0]):
  for j in range(desk.shape[1]):
    if warped_img[i][j]!=0:
      wrapped_HR[i][j]=warped_hp[i][j]
cv2.imshow("Homography with RANSAC", wrapped_HR)
cv2.waitKey(0)
#cv2.imwrite("./result/A2_hp_Homography with RANSAC_wrapped.png", wrapped_HR)

matches, kp1, kp2=match(d11, d10)

scaled_srcp, scaled_destp, Ts, Td=points(matches)

h=compute_homography_ransac(scaled_srcp, scaled_destp, 5)
warped_img=cv2.warpPerspective(d11, np.dot(np.dot(np.linalg.inv(Td), h), Ts), (d10.shape[1]*2, d10.shape[0]))

result=np.copy(warped_img)
result[0:d10.shape[0], 0:d10.shape[1]]=d10

for i in range(result.shape[1]):
  if result[-1][i]==0:
    length=i
    break
result=result[:, :length]
cv2.imshow("Stiched images", result)
cv2.waitKey(0)
#cv2.imwrite("./result/A2_Stitched images.png", result)


k=124
grad=900

left=np.zeros((result.shape[0], result.shape[1]))
for i in range(d10.shape[0]):
  for j in range(d10.shape[1]):
    left[i][j]=d10[i][j]

right=warped_img[:, :length].copy()
right[:, :grad]=0

mask=np.zeros((result.shape[0], k))
for i in range(k):
  mask[:, i]=(i)/k

Left=left.copy()
Right=right.copy()

Left[:, d10.shape[1]-k:d10.shape[1]]=left[:, d10.shape[1]-k:d10.shape[1]]*(1-mask)
Right[:, grad:grad+k]=right[:, grad:grad+k]*mask

Result=Left+Right
Result=Result.astype(np.uint8)
cv2.imshow("Stiched images", Result)
cv2.waitKey(0)
#cv2.imwrite("./result/A2_Stitched images with gradation.png", Result)