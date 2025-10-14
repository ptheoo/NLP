import gensim.downloader as api
import numpy as np
from typing import List, Tuple, Optional

# --- Hàm Tokenizer đơn giản (Thay thế cho Lab 1) ---
def simple_tokenizer(document: str) -> List[str]:
    """
    Một hàm tokenizer đơn giản: chuyển văn bản sang chữ thường và tách bằng khoảng trắng.
    """
    if not document:
        return []
    # Loại bỏ các dấu chấm câu đơn giản và chuyển thành chữ thường, sau đó tách
    cleaned_doc = document.lower().replace('.', '').replace(',', '').strip()
    return cleaned_doc.split()
# ---------------------------------------------


class WordEmbedder:
    """
    Tải mô hình nhúng từ (word embedding) được đào tạo trước và cung cấp các phương thức
    để truy xuất vector, tính toán độ tương đồng và nhúng tài liệu.
    """
    def __init__(self, model_name: str):
        """
        Tải mô hình được chỉ định từ kho dữ liệu của gensim.
        """
        print(f"Đang tải mô hình nhúng từ: {model_name}...")
        # Tải mô hình KeyedVectors
        self.model = api.load(model_name)
        print("Tải mô hình thành công.")
        # Lấy chiều kích thước của vector
        self.vector_dim = self.model.vector_size

    def get_vector(self, word: str) -> Optional[np.ndarray]:
        """
        Trả về vector nhúng cho một từ đã cho.
        Trả về None cho các từ nằm ngoài từ vựng (OOV - Out-of-Vocabulary).
        """
        try:
            return self.model[word]
        except KeyError:
            # Xử lý các từ OOV
            return None

    def get_similarity(self, word1: str, word2: str) -> float:
        """
        Trả về độ tương đồng cosine giữa các vector của hai từ.
        """
        try:
            return self.model.similarity(word1, word2)
        except KeyError:
            # Xử lý trường hợp một hoặc cả hai từ là OOV
            print(f"Cảnh báo: Một hoặc cả hai từ ('{word1}', '{word2}') không có trong từ vựng. Trả về 0.0.")
            return 0.0

    def get_most_similar(self, word: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Sử dụng phương thức most_similar có sẵn của mô hình để tìm N từ giống nhất.
        Trả về danh sách các bộ (từ, điểm_tương_đồng).
        """
        try:
            return self.model.most_similar(word, topn=top_n)
        except KeyError:
            # Xử lý từ OOV
            print(f"Cảnh báo: Từ '{word}' không có trong từ vựng.")
            return []

    # --- Task 3: Document Embedding (Nhúng Tài liệu) ---

    def embed_document(self, document: str) -> np.ndarray:
        """
        Tính toán vector tài liệu bằng cách lấy trung bình cộng các vector từ của
        tất cả các từ đã biết trong tài liệu.
        """
        # 1. Tokenize tài liệu
        tokens = simple_tokenizer(document)

        known_word_vectors = []
        for token in tokens:
            # 2. Lấy vector cho mỗi token. Bỏ qua các từ OOV.
            vector = self.get_vector(token)
            if vector is not None:
                known_word_vectors.append(vector)

        # 3. Nếu tài liệu không chứa từ nào đã biết, trả về vector 0 có đúng chiều kích thước
        if not known_word_vectors:
            print("Cảnh báo: Tài liệu không chứa từ nào trong từ vựng của mô hình. Trả về vector 0.")
            return np.zeros(self.vector_dim)

        # 4. Tính toán trung bình cộng theo từng phần tử (element-wise mean) của tất cả các vector từ
        # Chuyển danh sách vector thành ma trận NumPy, sau đó tính trung bình theo trục 0 (các hàng)
        document_matrix = np.stack(known_word_vectors)
        document_vector = np.mean(document_matrix, axis=0)

        return document_vector