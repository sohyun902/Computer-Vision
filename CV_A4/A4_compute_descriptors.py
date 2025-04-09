import numpy as np
import struct
np.random.seed(1)

# 각 features를 두번씩 load하기 때문에 대략 3분정도 시간이 소요됩니다.

class Kmeans:
  def __init__(self, data, k, max_iter=200):
    self.data=data
    self.k=k
    self.max_iter=max_iter

  def distance(self, f1, f2):
    dist=np.linalg.norm(f1-f2)
    return dist

  def init_centroid(self):
    idx=np.random.choice(self.data.shape[0], size=self.k, replace=False)
    centroids=self.data[idx]
    return centroids

  def create_cluster(self, centroids):
    min_dist=[]
    cluster=[]
    for i in range(self.data.shape[0]):
      feature=self.data[i]
      dist_list=[]
      for j in range(self.k):
        dist=self.distance(feature, centroids[j])
        dist_list.append(dist)

      min_idx=dist_list.index(min(dist_list))
      min_dist.append(min(dist_list))
      cluster.append(min_idx)

    cluster=np.array(cluster)
  
    return cluster

  def update(self):
    centroids=self.init_centroid()
    for i in range(self.max_iter):
      sum=np.zeros((self.k, self.data.shape[1]))
      count=np.zeros(self.k)
      cluster=self.create_cluster(centroids)
      for j in range(self.data.shape[0]):
        sum[cluster[j]]=sum[cluster[j]]+self.data[j]
        count[cluster[j]]=count[cluster[j]]+1
      for j in range(self.k):
        if count[j]!=0:
          centroids[j]=sum[j]/count[j]
        else:
          centroids[j]=self.data[np.random.choice(self.data.shape[0])]
    return centroids

# SIFT features 불러오기
data=[]
for i in range(2000):
  if i<10:
    n='000'+str(i)
  elif 10<=i<100:
    n='00'+str(i)
  elif 100<=i<1000:
    n='0'+str(i)
  else:
    n=str(i)
  with open('./features/sift/'+n+'.sift', 'rb') as f:
    byte=f.read()
    char_data=np.frombuffer(byte, dtype='uint8')
    for i in range(len(char_data)):
      data.append(int(char_data[i]))

data=np.array(data)
sift_data=data.reshape(-1, 128)

# SIFT features 랜덤 추출 및 클러스터링
num_cluster=32
"""sift_data_sample=sift_data[np.random.choice(sift_data.shape[0], size=200000, replace=False)]
kmeans=Kmeans(sift_data_sample, num_cluster)
sift_centroids=kmeans.update()

np.save('./SIFT_centroids.npy', sift_centroids)"""

# SIFT 센트로이드 파일 load
sift_centroids=np.load('./SIFT_centroids.npy')
sift_cluster=np.zeros(sift_data.shape[0])
for i in range(sift_data.shape[0]):
  sift_cluster[i]=np.argmin(np.linalg.norm(sift_data[i]-sift_centroids, axis=1))


# SIFT features vlad
sift_vlads=[]
start=0
end=0

for i in range(2000):

  if i<10:
    n='000'+str(i)
  elif 10<=i<100:
    n='00'+str(i)
  elif 100<=i<1000:
    n='0'+str(i)
  else:
    n=str(i)
  with open('./features/sift/'+n+'.sift', 'rb') as f:
    data=[]
    byte=f.read()
    char_data=np.frombuffer(byte, dtype='uint8')
    for i in range(len(char_data)):
      data.append(int(char_data[i]))

  data=np.array(data)
  data=data.reshape(-1, 128)
  start=end
  end=start+data.shape[0]
  idx=np.array(sift_cluster[start:end], dtype=int)
  res=data-sift_centroids[idx]
  sift_vlad=np.zeros((num_cluster, res.shape[1]))
  for j in range(res.shape[0]):
    sift_vlad[idx[j]]=sift_vlad[idx[j]]+res[j]

  sift_vlad=sift_vlad.flatten()
  sift_vlad=np.sign(sift_vlad)*np.sqrt(np.abs(sift_vlad))
  sift_vlad=sift_vlad/np.linalg.norm(sift_vlad)
  sift_vlads.append(sift_vlad)

sift_vlads=np.array(sift_vlads)


# CNN features 불러오기
data=[]
h=14
w=14
c=512

for i in range(2000):
  if i<10:
    n='000'+str(i)
  elif 10<=i<100:
    n='00'+str(i)
  elif 100<=i<1000:
    n='0'+str(i)
  else:
    n=str(i)
  with open('./features/cnn/'+n+'.cnn', 'rb') as f:
    byte=f.read()

  float_data=struct.unpack(f'{h*w*c}f', byte)
  data.append(float_data)

data=np.array(data)
cnn_data=data.reshape(-1, 512)

# CNN features 랜덤 추출 및 클러스터링
num_cluster=8
"""cnn_data_sample=cnn_data[np.random.choice(cnn_data.shape[0], size=80000, replace=False)]
kmeans=Kmeans(cnn_data_sample, num_cluster)
cnn_centroids=kmeans.update()

np.save('./CNN_centroids.npy', cnn_centroids)"""

# CNN 센트로이드 파일 load
cnn_centroids=np.load('./CNN_centroids.npy')
cnn_cluster=np.zeros(cnn_data.shape[0])
for i in range(cnn_data.shape[0]):
  cnn_cluster[i]=np.argmin(np.linalg.norm(cnn_data[i]-cnn_centroids, axis=1))

# CNN features vlad
cnn_vlads=[]
start=0
end=0

for i in range(2000):
  if i<10:
    n='000'+str(i)
  elif 10<=i<100:
    n='00'+str(i)
  elif 100<=i<1000:
    n='0'+str(i)
  else:
    n=str(i)
  with open('./features/cnn/'+n+'.cnn', 'rb') as f:
    b=f.read()

  data=struct.unpack(f'{h*w*c}f', b)

  data=np.array(data)
  data=data.reshape(-1, 512)
  start=end
  end=start+data.shape[0]
  idx=np.array(cnn_cluster[start:end], dtype=int)
  res=data-cnn_centroids[idx]
  cnn_vlad=np.zeros((num_cluster, res.shape[1]))
  for j in range(res.shape[0]):
    cnn_vlad[idx[j]]=cnn_vlad[idx[j]]+res[j]

  cnn_vlad=cnn_vlad.flatten()
  cnn_vlad=np.sign(cnn_vlad)*np.sqrt(np.abs(cnn_vlad))
  cnn_vlad=cnn_vlad/np.linalg.norm(cnn_vlad)

  cnn_vlads.append(cnn_vlad)
cnn_vlads=np.array(cnn_vlads)

# SIFT features로 구한 descriptor와 CNN features로 구한 descriptor 평균
des=(sift_vlads+cnn_vlads)/2

"""with open('./A4_SIFT.des', 'wb') as f:
    f.write(np.array(2000, dtype=np.int32).tobytes())
    f.write(np.array(4096, dtype=np.int32).tobytes())
    f.write(sift_vlads.astype('float32').tobytes())

with open('./A4_CNN.des', 'wb') as f:
    f.write(np.array(2000, dtype=np.int32).tobytes())
    f.write(np.array(4096, dtype=np.int32).tobytes())
    f.write(cnn_vlads.astype('float32').tobytes())"""

with open('./A4.des', 'wb') as f:
    f.write(np.array(2000, dtype=np.int32).tobytes())
    f.write(np.array(4096, dtype=np.int32).tobytes())
    f.write(des.astype('float32').tobytes())