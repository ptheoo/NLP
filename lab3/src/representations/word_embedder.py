import gensim.downloader as api
import numpy as np
from typing import List, Optional, Tuple

# --- Giả định về Tokenizer từ Lab 1 (Bạn hãy thay thế bằng code thực tế của mình) ---
def simple_tokenizer(document: str) -> List[str]:
    """Tách văn bản thành các từ và chuyển về chữ thường."""
    # Loại bỏ dấu câu và tách theo khoảng trắng
    import re
    tokens = re.findall(r'\b\w+\b', document.lower())
    return tokens
# -----------------------------------------------------------------------------------

class WordEmbedder:
    """
    Class để tải và khám phá các mô hình Word Embedding của Gensim.
    """
    def __init__(self, model_name: str):
        """
        Tải mô hình word embedding được chỉ định (e.g., 'glove-wiki-gigaword-50').
        """
        print(f"Đang tải mô hình: {model_name}...")
        try:
            # Model sẽ được lưu trong thuộc tính self.model
            self.model = api.load(model_name)
            print("Đã tải mô hình thành công.")
        except Exception as e:
            print(f"Lỗi khi tải mô hình {model_name}: {e}")
            self.model = None

    def get_vector(self, word: str) -> Optional[np.ndarray]:
        """
        Trả về vector embedding cho một từ.
        Trả về None nếu từ không có trong từ vựng (OOV).
        """
        if self.model is None:
            return None

        # Chuyển từ về chữ thường vì GloVe thường là case-sensitive
        word_lower = word.lower()
        try:
            # Truy cập trực tiếp vào mô hình hoạt động như một dictionary
            return self.model[word_lower]
        except KeyError:
            # Xử lý trường hợp Out-of-Vocabulary (OOV)
            return None

    def get_similarity(self, word1: str, word2: str) -> Optional[float]:
        """
        Trả về độ tương đồng cosine giữa hai từ.
        Trả về None nếu một trong hai từ là OOV.
        """
        if self.model is None:
            return None

        # Gensim Word2VecKeyedVectors có sẵn phương thức similarity
        word1_lower = word1.lower()
        word2_lower = word2.lower()
        
        # Kiểm tra OOV trước để tránh ngoại lệ
        if word1_lower not in self.model or word2_lower not in self.model:
             # Bạn có thể trả về một giá trị mặc định, nhưng None thì rõ ràng hơn
             return None

        try:
            return self.model.similarity(word1_lower, word2_lower)
        except KeyError:
            # Trường hợp lý thuyết, nhưng đã kiểm tra ở trên
            return None 

    def get_most_similar(self, word: str, top_n: int = 10) -> Optional[List[Tuple[str, float]]]:
        """
        Sử dụng phương thức most_similar() của mô hình để tìm N từ tương đồng nhất.
        """
        if self.model is None:
            return None
            
        word_lower = word.lower()
        if word_lower not in self.model:
            return None

        try:
            # Kết quả là list of (word, similarity) tuples
            return self.model.most_similar(word_lower, topn=top_n)
        except KeyError:
            # Xử lý OOV
            return None

# --------------------------------------------------------------------------------
# Task 3: Document Embedding được triển khai bên dưới
# --------------------------------------------------------------------------------

    def embed_document(self, document: str) -> Optional[np.ndarray]:
        """
        Biểu diễn tài liệu bằng cách tính trung bình vector của tất cả các từ trong tài liệu.
        """
        if self.model is None:
            return None

        # 1. Tách văn bản thành tokens (sử dụng Tokenizer từ Lab 1, ở đây là simple_tokenizer)
        tokens = simple_tokenizer(document)

        # 2. Lấy vector cho từng token, bỏ qua từ OOV
        word_vectors = []
        for token in tokens:
            vector = self.get_vector(token) # get_vector đã xử lý chữ thường và OOV
            if vector is not None:
                word_vectors.append(vector)
        
        # 3. Xử lý trường hợp không có từ nào được biết (empty document)
        if not word_vectors:
            # Trả về một vector zero với đúng chiều (dimension)
            # Lấy chiều từ vector của từ bất kỳ, hoặc giả định 50 nếu mô hình đã tải là 50-D
            if not self.model.vectors.size == 0:
                vector_dim = self.model.vector_size
            else:
                # Nếu mô hình trống (không nên xảy ra), giả định chiều
                vector_dim = 50 
            print(f"Cảnh báo: Tài liệu không chứa từ nào có trong từ vựng. Trả về vector zero {vector_dim}-D.")
            return np.zeros(vector_dim)

        # 4. Tính trung bình vector (element-wise mean)
        # np.mean có thể tính trung bình dọc theo trục 0 (trục của các vector)
        document_vector = np.mean(word_vectors, axis=0)

        return document_vector


if __name__ == '__main__':
    # --- Code thử nghiệm (Evaluation) có thể đặt tại đây hoặc trong file test riêng ---
    # Tôi sẽ đặt nó trong file test riêng theo yêu cầu.
    pass