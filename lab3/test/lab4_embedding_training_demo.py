import os
import logging
from gensim.models import Word2Vec
from typing import List, Iterator

# Cấu hình logging để thấy thông báo tiến trình huấn luyện
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# --- THIẾT LẬP ĐƯỜNG DẪN ---

# Xác định thư mục hiện tại của script (test/)
DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
# Xác định thư mục gốc của dự án (lab3)
PROJECT_ROOT = os.path.dirname(DEMO_DIR) 
# Xác định thư mục NLP 
NLP_ROOT = os.path.dirname(PROJECT_ROOT)

# Đường dẫn đến file dữ liệu (data nằm ngang hàng với lab3):
DATA_PATH = os.path.join(NLP_ROOT, 'data', 'UD_English-EWT', 'en_ewt-ud-train.txt')

# Đường dẫn thư mục results (nằm trong lab3):
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
MODEL_PATH = os.path.join(RESULTS_DIR, 'word2vec_ewt.model')

# Đảm bảo thư mục results tồn tại
os.makedirs(RESULTS_DIR, exist_ok=True)
# ----------------------------------------------------

# --- Class Streaming Dữ liệu ---

class MyCorpus:
    """
    Iterator đọc dữ liệu từ file, xử lý từng dòng (câu) một cách hiệu quả bộ nhớ.
    """
    def __init__(self, path_to_data: str):
        self.path_to_data = path_to_data

    def __iter__(self) -> Iterator[List[str]]:
        """
        Lặp qua từng dòng, chuyển về chữ thường và tách từ.
        """
        sentences_count = 0
        try:
            with open(self.path_to_data, 'r', encoding='utf-8') as f:
                for line in f:
                    # Bỏ qua dòng trống và dòng comment (bắt đầu bằng #)
                    if not line.strip() or line.startswith('#'):
                        continue
                    
                    # Giả định đơn giản: chuyển về chữ thường và tách bằng khoảng trắng
                    yield line.lower().strip().split()
                    sentences_count += 1
        except FileNotFoundError:
            raise FileNotFoundError(f"\nLỖI: Không tìm thấy file dữ liệu tại: {self.path_to_data}\nKiểm tra lại tên file và cấu trúc thư mục NLP/data/...")
        
        print(f"Tổng số câu được đọc để huấn luyện: {sentences_count}")


# --- Hàm chính để Huấn luyện và Demo ---

def train_and_demo_word2vec():
    """
    Thực hiện: Stream Data -> Train Model -> Save Model -> Demonstrate Usage.
    """
    print("="*70)
    print("BẮT ĐẦU: HUẤN LUYỆN MÔ HÌNH WORD2VEC TỪ ĐẦU")
    print("="*70)
    
    # 1. Streams Data
    print(f"\n1. Đang tải và xử lý dữ liệu từ: {DATA_PATH}...")
    sentences = MyCorpus(DATA_PATH)

    # 2. Trains a Model
    print("\n2. Bắt đầu huấn luyện mô hình Word2Vec (kích thước 100, 10 epochs)...")
    # Thông số được chọn để mô hình có thể học được gì đó trên dữ liệu nhỏ
    model = Word2Vec(
        sentences=sentences,
        vector_size=100,      # Chiều kích thước vector (thường dùng 100)
        window=5,             # Kích thước cửa sổ
        min_count=5,          # Chỉ giữ lại các từ xuất hiện ít nhất 5 lần
        workers=4,            # Số luồng xử lý
        epochs=50,           # Số lần lặp huấn luyện
        sg=1
    )
    print("Huấn luyện mô hình Word2Vec hoàn tất.")
    print(f"Kích thước từ vựng mô hình (vocab size): {len(model.wv)}")

    # 3. Saves the Model
    print(f"\n3. Đang lưu mô hình đã huấn luyện tại: {MODEL_PATH}")
    model.save(MODEL_PATH)
    print("Lưu mô hình thành công.")
    
    wv = model.wv

    # 4. Demonstrates Usage
    print("\n4. Demo sử dụng mô hình Word2Vec đã huấn luyện:")
    
    # --- Demo A: Tương đồng (Similarity) ---
    test_word = 'student' # Chọn từ phổ biến hơn để đảm bảo có trong vocab
    print(f"\n   A. 10 từ tương đồng nhất với '{test_word}':")
    try:
        if test_word not in wv:
             print(f"      Lỗi: Từ '{test_word}' không có trong từ vựng ({len(wv)} từ).")
        else:
            similar_words = wv.most_similar(test_word)
            for i, (word, score) in enumerate(similar_words):
                print(f"      {i+1}. {word}: {score:.4f}")
    except KeyError:
        # Should not happen if checked above, but good practice
        print(f"      Lỗi: Không thể tìm từ '{test_word}'.")
        
    # --- Demo B: Tương tự (Analogy) ---
    # Phép toán: king - man + woman = ? (mong đợi: queen)
    positive_words = ['woman', 'king']
    negative_words = ['man']
    
    print(f"\n   B. Giải quyết bài toán tương tự: {positive_words[1]} - {negative_words[0]} + {positive_words[0]} = ?")
    
    try:
        # Kiểm tra xem tất cả các từ có tồn tại trong từ vựng không
        all_words_exist = all(word in wv for word in positive_words + negative_words)
        
        if all_words_exist:
            analogy_result = wv.most_similar(positive=positive_words, negative=negative_words, topn=3)
            print("      Kết quả (Top 3):")
            for i, (word, score) in enumerate(analogy_result):
                print(f"      {i+1}. {word} (Score: {score:.4f})")
        else:
            missing_words = [w for w in positive_words + negative_words if w not in wv]
            print(f"      Lỗi: Không đủ từ vựng để chạy Analogy. Thiếu: {missing_words}")

    except Exception as e:
        print(f"      Lỗi khi chạy Analogy: {e}")
        
    print("\n" + "="*70)

if __name__ == '__main__':
    train_and_demo_word2vec()