# python version update 필요
# import module
import tensorflow
from keras import layers, models
import keras
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
from time import time
from glob import glob
import re
from typing import Final
from tqdm import tqdm
import os
import csv
# 함수 정의
def read_file(file_name):
    # 파일 가져와야함
    title = pd.read_csv(file_name, nrows = 1).columns.tolist()
    comment = pd.read_csv(file_name, nrows = 1, skiprows =1).columns.tolist()

    data_num = int(re.findall(r'\d+', comment[0])[0])
    data = pd.read_csv(file_name, header=None, nrows = data_num, skiprows = 2, sep ='\t').to_numpy()

    # data cleasing
    x,y,z = data[:,0], data[:,1], data[:,2]
    u,v,w,p = data[:, 3], data[:,4], data[:,5], data[:,6]
    
    # x,y,z 좌표의 개수
    Nx, Ny, Nz = len(set(x)), len(set(y)), len(set(z))
    # dataset의 기준 좌표계와 numpy 좌표계의 차이가 있기 때문에 변환이 필요함

    u = u.reshape((Nx, Nz, Ny), order = 'F')
    # transpose 는 행렬곱을 해주기위함
    u = np.transpose(u, (2,1,0))
   
    v = v.reshape((Nx, Nz, Ny), order = 'F')
    # transpose 는 행렬곱을 해주기위함
    v = np.transpose(v, (2,1,0))
    
    w = w.reshape((Nx, Nz, Ny), order = 'F')
    # transpose 는 행렬곱을 해주기위함
    w = np.transpose(w, (2,1,0))
    
    p = p.reshape((Nx, Nz, Ny), order = 'F')
    # transpose 는 행렬곱을 해주기위함
    p = np.transpose(p, (2,1,0))

    return (u, v, w, p, Nx, Ny, Nz)
def norm_data(data, vmin, vmax): # -1 ~ 1
    # 정규 분포로 만들고자 함.
    alpha = 2 / (vmax - vmin)
    norm_data = alpha * (data - vmin) - 1 
    return norm_data
def norm_data2(data, vmin, vmax):
    # 교수님이 만드신것
    alpha = 2 / (vmax - vmin)
    beta = -1 - alpha * vmin
    norm_data = alpha * data + beta
    return norm_data
def denorm_data(norm_data,vmin,vmax):
    alpha = 2/(vmax-vmin)
    data = (norm_data+1) / alpha + vmin
    return data
def plot_image(outfile,data,vmin,vmax): 
#outfile: 저장파일명, data: 2차원 데이터 행렬, vmin: colormap 최소값, vmax: colormap 최대값
    vel = re.findall(r"vel_(\d+\.\d+|\d+|\d+)", outfile)[0] # 파일명에서 경계속도값 추출
    filename = os.path.basename(outfile)
    filename = os.path.splitext(filename)[0]
    variable = filename[0]
    data_type = filename[2]
    start=time.time()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(data, cmap='jet', vmin=vmin,vmax=vmax)
    plt.tight_layout()
    plt.annotate('Initial U-Vel= ' + vel,xy=(0.05, 0.95), xycoords='axes fraction', fontsize=16)
    plt.annotate('Variable = ' + variable,xy=(0.05, 0.90), xycoords='axes fraction', fontsize=16)
    if data_type == 'i':
        plt.annotate('Initial Field',xy=(0.05, 0.85), xycoords='axes fraction', fontsize=16)
    elif data_type == 'p':
        plt.annotate('Pred. Result',xy=(0.05, 0.85), xycoords='axes fraction', fontsize=16)
    elif data_type == 'G':
        plt.annotate('Time-averaged GT Field',xy=(0.05, 0.85), xycoords='axes fraction', fontsize=16)
    else:
        pass

    fig.colorbar(im, orientation="vertical")
    plt.savefig(outfile)
    plt.close('all')
    end=time.time()
    dT = end-start
def permutation_train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1004):

    import numpy as np

    

    test_num = int(X.shape[0] * test_size)

    train_num = X.shape[0] - test_num

    

    if shuffle:

        np.random.seed(random_state)

        shuffled = np.random.permutation(X.shape[0])

        X = X[shuffled,:]

        y = y[shuffled]

        X_train = X[:train_num]

        X_test = X[train_num:]

        y_train = y[:train_num]

        y_test = y[train_num:]

    else:

        X_train = X[:train_num]

        X_test = X[train_num:]

        y_train = y[:train_num]

        y_test = y[train_num:]

        

    return X_train, X_test, y_train, y_test

# permutation_train_test_split() reference : https://rfriend.tistory.com/519

# Conv2d -> Activation function -> Pooling 
# input_size = (46, 185, 127)
# tf.keras.layers.Conv2D(
#     filters,
#     kernel_size,
#     strides=(1, 1),
#     padding='valid',
#     data_format=None,
#     dilation_rate=(1, 1),
#     groups=1,
#     activation=None,
#     use_bias=True,
#     kernel_initializer='glorot_uniform',
#     bias_initializer='zeros',
#     kernel_regularizer=None,
#     bias_regularizer=None,
#     activity_regularizer=None,
#     kernel_constraint=None,
#     bias_constraint=None,
#     **kwargs
# )

# Pre-processing 

TRAIN_DIR_NAME:Final = "/Users/bangseongjin/Downloads/ddfe/urban/Base_code/data/Train"
TEST_DIR_NAME:Final = "/Users/bangseongjin/Downloads/ddfe/urban/Base_code/data/Test"

train_start_time = time()

train_initial_file = sorted(glob(TRAIN_DIR_NAME + '/initial' + '/*.dat'))
train_mean_file = sorted(glob(TRAIN_DIR_NAME + '/mean' + '/*.dat'))

train_initial_u, train_mean_u = [], []
train_initial_v, train_mean_v = [], []
train_initial_w, train_mean_w = [], []
train_initial_p, train_mean_p = [], []

for i in tqdm(range(len(train_initial_file))):
    
    # initial_value
    u,v,w,p,_,_,_ = read_file(train_initial_file[i])
    train_initial_u.append(u)
    train_initial_v.append(v)
    train_initial_w.append(w)
    train_initial_p.append(p)

    # mean_value
    u,v,w,p,Nx,Ny,Nz = read_file(train_mean_file[i])
    train_mean_u.append(u)
    train_mean_v.append(v)
    train_mean_w.append(w)
    train_mean_p.append(p)

train_mean_u, train_initial_u = np.array(train_mean_u), np.array(train_initial_u)
train_mean_v, train_initial_v = np.array(train_mean_v), np.array(train_initial_v)
train_mean_w, train_initial_w = np.array(train_mean_w), np.array(train_initial_w)
train_mean_p, train_initial_p = np.array(train_mean_p), np.array(train_initial_p)


# mean file로 하는 이유는 mean 값의 u 만 변동하기 때문
# ex) 0.5의 경계속도를 지니는 initial_file 에서 min, max 를 추출하면 0.5 따라서 normalize 할 수 없음
umin, umax = np.min(train_mean_u), np.max(train_mean_u)
vmin, vmax = np.min(train_mean_v), np.max(train_mean_v)
wmin, wmax = np.min(train_mean_w), np.max(train_mean_w)
pmin, pmax = np.min(train_mean_p), np.max(train_mean_p)

# 학습 데이터 min-max normalization
train_initial_u, train_mean_u = norm_data(train_initial_u, umin, umax ), norm_data(train_mean_u, umin, umax)
train_mean_v, train_initial_v = norm_data(train_mean_v, vmin, vmax), norm_data(train_initial_v, vmin, vmax )
train_mean_w, train_initial_w = norm_data(train_mean_w, wmin, wmax), norm_data(train_initial_w, wmin, wmax )
train_mean_p, train_initial_p = norm_data(train_mean_p, pmin, pmax), norm_data(train_initial_p, pmin, pmax )

X_train = train_initial_u[:9, :, :, 3]
X_test = train_mean_p[9:, :, :, 3]

y_train = train_initial_u[:9, :, :, 3]
y_test = train_mean_p[9:, :, :, 3]

# transpose(2,1,0) 안해도 된다.
# model.summary에서 size 확인하기
# X_train, X_test, y_train, y_test = permutation_train_test_split(train_initial_u, train_mean_u, test_size = 0.2, shuffle = True, random_state = 1004)
print(X_train.shape)
print(y_train.shape)
# input_size 
# (N - Fitter_size + 2padding) / Stride + 1
input_size = (46, 185, 1)

''''''
model = models.Sequential() # layer 를 이을 것 
model.add(layers.Conv2D(4, (2, 2), padding = 'same', activation='leaky_relu', input_shape= input_size))
model.add(layers.Conv2D(16, (2, 2), padding = 'same', activation='leaky_relu')) 
# model.add(layers.AvgPool3D(pool_size = (2,2,2)))

model.add(layers.Conv2D(16, (2, 2), padding = 'same', activation='leaky_relu'))
model.add(layers.Conv2D(16, (2, 2), padding = 'same', activation='leaky_relu')) 
# model.add(layers.AvgPool3D(pool_size = (2,2,2)))
 
model.add(layers.Conv2D(16, (2, 2), padding = 'same', activation='leaky_relu'))
model.add(layers.Conv2D(16, (2, 2), padding = 'same', activation='leaky_relu')) 
# model.add(layers.GlobalAveragePooling3D())

# model.add(layers.Flatten())
# model.add(layers.Dense(27680, activation='linear'))
# model.add(layers.Dense(27680, activation='linear'))

model.add(layers.Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='linear'))
# model.add(layers.Reshape(input_size))
'''
model.add(layers.Flatten())
model.add(layers.Dense(27680, activation='linear'))
model.add(layers.Dense(27680, activation='linear'))

model.add(layers.Conv3D(filters=1, kernel_size=(1, 1, 1), padding='same', activation='linear'))
model.add(layers.Reshape(input_size))
'''
model.summary()

# pooling 제외 # average pooling 고민 : 장점 : 음의 영역이 많으므로 max pooling을 사용시에 영역간의 큰 편차가 생길 수 있을
# 단점 : 초기 속도에 따른 큰 변화가 없을 수 있음 , small amount of translation
# reference : https://paperswithcode.com/method/average-pooling
# summary에 None 이 나오는 이유 : dataset len

# CategoricalCrossentropy

model.compile(optimizer = 'Adam', loss = keras.losses.MeanSquaredError(), metrics = ['accuracy'])
history = model.fit(X_train, y_train, epochs = 100)


# 모델 평가
prediction =  model.predict(X_test)
print(prediction.shape)
prediction = np.reshape(prediction, (46, 185))
print(type(prediction))
print(prediction)

filename = 'test_case2_vel_4.25_data_file_set_000000000'
plot_image('u_CNN_'+filename+'.png', denorm_data(prediction,umin,umax),umin,umax)