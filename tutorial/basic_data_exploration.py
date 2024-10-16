import pandas as pd
from sklearn.tree import DecisionTreeRegressor

melbourne_file_path = 'melb_data.csv'

melbourne_data = pd.read_csv(melbourne_file_path)

melbourne_data.dropna(axis=0) # axis인 경우 NA가 존재하는 행 제거, 1이면 열 제거

y = melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude', ]

x = melbourne_data[melbourne_features]
# print(x.describe()) 이 상태로 출력하면 위에서 명시한 행들만 데이터로 반환된다.
# print(x.head()) 데이터 순서상 가장 위, default = 5

melbourne_model = DecisionTreeRegressor(random_state=1) #의사결정 트리 회귀 모델, random_state로 트리 분할시 랜덤성
melbourne_model.fit(x, y) # 독립변수, 종속변수 순서로 입력하며 독립변수에 따른 종속변수값을 학습한다.

# 아래의 로직을 통해 보는 것 : x의 가장 앞 5개를 구한 후 학습을 통해 예측한 회귀모델이 비어있는 가격 컬럼에 어떤 데이터가 나올지 학습 후 알려준다.
print("Making predictions for the following 5 houses:")
print(x.head())
print("The predictions are")
print(melbourne_model.predict(x.head()))