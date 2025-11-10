from typing import List, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Để Vectorizer có thể là TfidfVectorizer hoặc CountVectorizer
class Vectorizer:
    def fit_transform(self, texts):
        raise NotImplementedError
    def transform(self, texts):
        raise NotImplementedError

class TextClassifier:
    """
    Bộ phân loại văn bản sử dụng Logistic Regression và Vectorizer ngoài.
    """
    def __init__(self, vectorizer: Vectorizer):
        """
        Khởi tạo bộ phân loại.
        """
        self.vectorizer = vectorizer
        self._model = None
        
    def fit(self, texts: List[str], labels: List[int]):
        """
        Huấn luyện bộ phân loại.
        """
        # 1. Vector hóa dữ liệu huấn luyện (fit_transform)
        X = self.vectorizer.fit_transform(texts)
        
        # 2. Khởi tạo và huấn luyện mô hình
        # Sử dụng solver='liblinear' cho các tập dữ liệu nhỏ.
        self._model = LogisticRegression(solver='liblinear', random_state=42)
        self._model.fit(X, labels)
        print("Mô hình Logistic Regression đã được huấn luyện.")

    def predict(self, texts: List[str]) -> List[int]:
        """
        Thực hiện dự đoán nhãn trên văn bản mới.
        """
        # 1. Vector hóa dữ liệu mới (chỉ transform)
        if self._model is None:
            raise Exception("Mô hình chưa được huấn luyện.")
            
        X = self.vectorizer.transform(texts)
        
        # 2. Dự đoán
        predictions = self._model.predict(X)
        return predictions.tolist()

    def evaluate(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        """
        Tính toán các chỉ số đánh giá.
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        return metrics