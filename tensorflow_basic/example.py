import tensorflow as tf

heights = [ 170, 180, 175, 160]
foot_sizes = [ 260, 270, 265, 255]

# 기본적으로 하나의 값으로 하나의 결과를 예측하기 위해서는 1차 방정식을 이용한다. 
# foot_size = height * a + b

a = tf.Variable(0.1)
b = tf.Variable(0.2)

def loss_function(foot_size, height): 
    return tf.square(foot_size - (a * height + b) )

# 경사하강법으로 a, b를 학습해 업데이트해준다. Adam이라는 모델로 경사하강법을 통해 변수들을 업데이트해준다. 
optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

# minimize로 업데이트를 실행시킨다. 변수로는 손실함수 & 업데이트시킬 변수들을 넣어야 한다. 
for i in range (300): 
    for index in range(len(heights)):
        optimizer.minimize(loss_function(heights[index], foot_sizes[index]),[a, b])

print(a.numpy())
print(b.numpy())