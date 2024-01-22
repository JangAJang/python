import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

num_samples = int(input("입력할 알고리즘 문제의 개수를 입력하세요: "))
problem_descriptions = []
codes = []
algorithm_categories = []
solution_approaches = []

nltk.download('punkt')

for i in range(num_samples):
    print(f"\n샘플 #{i + 1}")
    
    # 터미널에서 문제 내용 입력 받기
    problem_description = input("알고리즘 문제 내용을 입력하세요: ")
    problem_descriptions.append(problem_description)

    # 터미널에서 코드 입력 받기
    code = input("해당 문제에 대한 코드를 입력하세요: ")
    codes.append(code)

    # 터미널에서 알고리즘 카테고리 입력 받기
    algorithm_category = input("알고리즘 카테고리를 입력하세요: ")
    algorithm_categories.append(algorithm_category)

    # 터미널에서 문제 해결 방식 입력 받기
    solution_approach = input("문제 해결 방식을 입력하세요: ")
    solution_approaches.append(solution_approach)


# 문제 내용 및 코드 토큰화
problem_description_tokens = word_tokenize(problem_description)
code_tokens = word_tokenize(code)


# TF-IDF를 사용한 특징 추출
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(problem_descriptions + codes)
y_algorithm_category = algorithm_categories
y_solution_approach = solution_approaches

# 학습 데이터와 테스트 데이터로 분리
X_train, X_test, y_train_algorithm, y_test_algorithm, y_train_solution, y_test_solution = train_test_split(X, y_algorithm_category, y_solution_approach, test_size=0.2, random_state=42)

# 다항 나이브 베이즈 분류 모델 학습
model_algorithm = MultinomialNB()
model_algorithm.fit(X_train, y_train_algorithm)

# 다른 모델을 사용하여 문제 해결 방식 학습
model_solution = ...
model_solution.fit(X_train, y_train_solution)

# 새로운 알고리즘 문제에 대한 최적의 코드 예측
new_problem_description = "새로운 문제입니다."
new_code = "def solution(): ..."
new_problem_description_vectorized = vectorizer.transform([new_problem_description])
new_code_vectorized = vectorizer.transform([new_code])

predicted_algorithm_category = model_algorithm.predict(new_problem_description_vectorized)
predicted_solution_approach = model_solution.predict(new_code_vectorized)

print("Predicted Algorithm Category: {predicted_algorithm_category}")
print("Predicted Solution Approach: {predicted_solution_approach}")

