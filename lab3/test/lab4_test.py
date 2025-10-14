import numpy as np
# Giả định đường dẫn từ root project đến src/representations là chính xác
from src.representations.word_embedder import WordEmbedder 
import time

def run_evaluation():
    # Model name theo yêu cầu
    model_name = 'glove-wiki-gigaword-50'
    
    # 1. Khởi tạo WordEmbedder (sẽ tải mô hình)
    start_time = time.time()
    embedder = WordEmbedder(model_name)
    end_time = time.time()
    print("-" * 50)
    print(f"Thời gian tải mô hình: {end_time - start_time:.2f} giây")
    
    if embedder.model is None:
        print("Không thể tiếp tục vì mô hình không được tải.")
        return

    # --- Thực hiện và in kết quả các thao tác ---

    # 1. Lấy vector cho từ 'king'
    word_king = 'king'
    vector_king = embedder.get_vector(word_king)
    print("\n## 1. Vector cho từ 'king'")
    if vector_king is not None:
        print(f"Từ: '{word_king}'")
        print(f"Kích thước vector: {vector_king.shape}")
        # In 5 giá trị đầu tiên để kiểm tra
        print(f"Vector (5 phần tử đầu): {vector_king[:5]}")
    else:
        print(f"Từ '{word_king}' không có trong từ vựng.")
    
    print("-" * 50)

    # 2. Lấy độ tương đồng giữa 'king' và 'queen', và giữa 'king' và 'man'
    word_queen = 'queen'
    word_man = 'man'
    
    sim_king_queen = embedder.get_similarity(word_king, word_queen)
    sim_king_man = embedder.get_similarity(word_king, word_man)
    
    print("## 2. Độ tương đồng Cosine")
    print(f"Độ tương đồng giữa '{word_king}' và '{word_queen}': {sim_king_queen:.4f}" if sim_king_queen is not None else "OOV")
    print(f"Độ tương đồng giữa '{word_king}' và '{word_man}': {sim_king_man:.4f}" if sim_king_man is not None else "OOV")

    print("-" * 50)

    # 3. Lấy 10 từ tương đồng nhất với 'computer'
    word_computer = 'computer'
    top_n = 10
    most_similar_words = embedder.get_most_similar(word_computer, top_n)
    
    print(f"## 3. {top_n} từ tương đồng nhất với '{word_computer}'")
    if most_similar_words:
        for i, (word, similarity) in enumerate(most_similar_words):
            print(f"  {i+1}. {word:<15} (Tương đồng: {similarity:.4f})")
    else:
        print(f"Từ '{word_computer}' không có trong từ vựng.")

    print("-" * 50)

    # 4. Biểu diễn câu "The queen rules the country."
    document = "The queen rules the country."
    document_vector = embedder.embed_document(document)
    
    print(f"## 4. Biểu diễn tài liệu (Document Embedding) cho câu: '{document}'")
    if document_vector is not None:
        print(f"Kích thước vector tài liệu: {document_vector.shape}")
        # In 5 giá trị đầu tiên
        print(f"Vector tài liệu (5 phần tử đầu): {document_vector[:5]}")
    else:
        print("Không thể tạo vector tài liệu.")

if __name__ == '__main__':
    run_evaluation()