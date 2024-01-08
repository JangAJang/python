import pandas as pd

# csv를 pandas를 이용해 가져온다. 
datas = pd.read_csv('gpascore.csv')

# 데이터를 전처리해준다. 
# 비어있는 값이 있을 때 행을 삭제하거나, 평균값을 넣어준다. 
# print(datas.isnull().sum())
# admit    0
# gre      1
# gpa      0
# rank     0
# 비어있는 컬럼을 알려준다. 
datas = datas.dropna() # 빈 로우 삭제
result = datas['admit'].values
inputs = []

for i, rows in datas.iterrows() : 
    inputs.append([rows['gre'], rows['gpa'], rows['rank']])

import tensorflow as tf
import numpy as np

# tf.keras.models.Sequential([])배열 안에 레이어를 만들면 모델을 만들 수 있다. 
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='tanh'), 
    # 레이어를 만들어주는 방식이다. 노드의 개수를 괄호에 담아준다. 괄호 안의 수는 임의로 넣을 수 있지만, 2의 거듭제곱으로 담는 것이 관용적이다. 
    # 노드의 수 이후에 활성함수를 넣어주어야 한다. 
    # sigmoid, tanh, softmax, leakyRelu등이 존재한다. 
    tf.keras.layers.Dense(128, activation='tanh'), 
    tf.keras.layers.Dense(1, activation='sigmoid'), # 여기에서는 최종적인 확률이 나와야 한다. 0~1이어야 하며, sigmoid는 이렇게 만들어준다. 
])

# compile을 통해 해당 모델을 컴파일할 수 있다. 
# loss function은 다양한 종류가 있다. 
# binary_crossentropy는 결과가 0과 1사이의 분류/확률문제에서 쓰인다. 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#fit함수를 통해 학습을 진행한다. 
#변수는 입력해야할 데이터, 실제 결과값, 반복횟수(epoch=)로 담는다. 
# inputs, result는 단순 배열로 넣으면 안되고, nunpy혹은 Tensor로 넣어야 한다. 
model.fit( np.array(inputs), np.array(result), epochs=2000)

# 예측
# 최적의 w값을 찾았을 것이다. 
print(model.predict([[750, 3.70, 3], [400, 2.2, 1]]))