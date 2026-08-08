# algorithm_solver

풀었던 알고리즘 문제를 카테고리별로 분류해주는 실험용 스크립트.

- 진입점: `categorizer.py` — 터미널 입력(문제 설명, 코드, 카테고리, 풀이 방식)을 받아 TF-IDF + Naive Bayes로 카테고리를 분류/학습
- 핵심 파일:
  - `categorizer.py` — nltk 토큰화 + scikit-learn TfidfVectorizer/MultinomialNB 사용
- 연관: `leetcode/`에 쌓인 풀이를 분류 대상으로 삼는 용도로 만들어짐 (직접 연동은 없음, 수동 입력)

<!-- TODO(확인 필요): nltk punkt 데이터, sklearn 등 의존성이 .venv에 설치돼 있는지 확인 필요 -->
