import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.representations.word_embedder import WordEmbedder

# Định nghĩa mô hình sử dụng
MODEL_NAME = 'glove-wiki-gigaword-50'

def run_evaluation():
    """
    Khởi tạo WordEmbedder và thực hiện các thao tác được yêu cầu.
    """
    # Khởi tạo WordEmbedder
    embedder = WordEmbedder(MODEL_NAME)

    print("\n" + "="*50)
    print("ĐÁNH GIÁ: NHÚNG TỪ VÀ NHÚNG TÀI LIỆU")
    print("="*50)

    # 1. Lấy vector cho từ 'king'
    word_king = 'king'
    vector_king = embedder.get_vector(word_king)
    print(f"1. Vector cho từ '{word_king}':")
    if vector_king is not None:
        print(f"   Kích thước (Shape): {vector_king.shape}")
        # In một vài phần tử để kiểm tra
        print(f"   Các phần tử mẫu (5 phần tử đầu): {vector_king[:5]}")
    else:
        print(f"   Từ '{word_king}' là OOV.")

    print("-" * 50)

    # 2. Lấy độ tương đồng giữa 'king' và 'queen', và giữa 'king' và 'man'
    word_queen = 'queen'
    word_man = 'man'
    sim_king_queen = embedder.get_similarity(word_king, word_queen)
    sim_king_man = embedder.get_similarity(word_king, word_man)

    print(f"2. Độ Tương đồng Cosine (Cosine Similarity):")
    print(f"   Similarity('{word_king}', '{word_queen}'): {sim_king_queen:.4f}")
    print(f"   Similarity('{word_king}', '{word_man}'):   {sim_king_man:.4f}")

    print("-" * 50)

    # 3. Lấy 10 từ giống nhất với 'computer'
    word_comp = 'computer'
    top_n = 10
    most_similar_comp = embedder.get_most_similar(word_comp, top_n)

    print(f"3. Top {top_n} từ giống nhất với '{word_comp}':")
    for i, (word, score) in enumerate(most_similar_comp):
        print(f"   {i+1}. {word}: {score:.4f}")

    print("-" * 50)

    # 4. Nhúng câu “The queen rules the country.” và in vector tài liệu thu được
    sentence = "The queen rules the country."
    document_vector = embedder.embed_document(sentence)

    print(f"4. Nhúng Tài liệu (Trung bình cộng các Vector Từ):")
    print(f"   Câu: \"{sentence}\"")
    print(f"   Kích thước Vector Tài liệu (Shape): {document_vector.shape}")
    # In một vài phần tử để kiểm tra
    print(f"   Các phần tử mẫu (5 phần tử đầu): {document_vector[:5]}")
    # Kiểm tra xem có phải là vector 0 không
    is_zero = np.allclose(document_vector, np.zeros_like(document_vector))
    print(f"   Có phải là vector 0? {'Có' if is_zero else 'Không'}")

    print("\n" + "="*50)

if __name__ == '__main__':
    run_evaluation()