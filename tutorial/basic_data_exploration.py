import sys

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

melbourne_file_path = 'melb_data.csv'

melbourne_data = pd.read_csv(melbourne_file_path)

melbourne_data.dropna(axis=0) # axis인 경우 NA가 존재하는 행 제거, 1이면 열 제거

y = melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

x = melbourne_data[melbourne_features]
# print(x.describe()) 이 상태로 출력하면 위에서 명시한 행들만 데이터로 반환된다.
# print(x.head()) 데이터 순서상 가장 위, default = 5

melbourne_model = DecisionTreeRegressor(random_state=1) #의사결정 트리 회귀 모델, random_state로 트리 분할시 랜덤성
melbourne_model.fit(x, y) # 독립변수, 종속변수 순서로 입력하며 독립변수에 따른 종속변수값을 학습한다.

# 아래의 로직을 통해 보는 것 : x의 가장 앞 5개를 구한 후 학습을 통해 예측한 회귀모델이 비어있는 가격 컬럼에 어떤 데이터가 나올지 학습 후 알려준다.
# print("Making predictions for the following 5 houses:")
# print(x.head())
# print("The predictions are")
# print(melbourne_model.predict(x.head()))

# 예측결과의 오차값 계산
predicted_home_prices = melbourne_model.predict(x)
print(mean_absolute_error(y, predicted_home_prices))

# 트레이닝 데이터 및 검증 데이터 분리해서 사용
train_X, val_X, train_y, val_y = train_test_split(x, y, random_state = 0)
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))

# 과적합, 부적합을 피하기 위한 첫 단계, mae를 구하기
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

# 각 최대 리프노드당 MAE 구하기 : 리프노드 수가 많아질 수록 과적합에 가까워진다. 적을수로 부적합에 가까워진다.

def get_optimized_leaf_nodes_count(train_X, val_X, train_y, val_y):
    left = 5
    right = melbourne_data.shape[0]
    min_mae = 100000000000000
    best_leaf_nodes = left

    while left <= right:
        mid = (left + right) // 2
        mae = get_mae(mid, train_X, val_X, train_y, val_y)

        if mae < min_mae:
            min_mae = mae
            best_leaf_nodes = mid

        # 이분 탐색 진행: mae가 감소하는 방향을 기준으로 탐색 범위를 줄임
        if get_mae(mid - 1, train_X, val_X, train_y, val_y) < mae:
            right = mid - 1
        else:
            left = mid + 1

    return best_leaf_nodes, min_mae

# 최적의 리프 노드 수와 그에 따른 최소 MAE 출력
best_leaf_nodes, min_mae = get_optimized_leaf_nodes_count(train_X, val_X, train_y, val_y)
print(f"Best leaf nodes: {best_leaf_nodes}, Minimum MAE: {min_mae}")

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))