import os
import re
import logging
from pyspark.sql import SparkSession
from pyspark.ml.feature import Word2Vec
from pyspark.sql.functions import col, lower, regexp_replace, split

# Cấu hình logging cơ bản
logging.basicConfig(level=logging.INFO)

# --- THIẾT LẬP ĐƯỜNG DẪN DỮ LIỆU ---
# Giả định file JSON nằm trong thư mục data/ của dự án
# Lấy đường dẫn gốc của dự án (lab3)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Tên file dữ liệu
DATA_FILE_NAME = "c4-train.00000-of-01024-30K.json"
# Giả sử file nằm ở một thư mục data ngang hàng với lab3
DATA_PATH = os.path.join(os.path.dirname(PROJECT_ROOT), 'data', DATA_FILE_NAME)

# Nếu bạn chắc chắn file nằm trong thư mục 'data' CỦA lab3, hãy dùng dòng này thay thế:
# DATA_PATH = os.path.join(PROJECT_ROOT, 'data', DATA_FILE_NAME) 

# --- HÀM CHÍNH ---

def main():
    print("="*80)
    print("BẮT ĐẦU: HUẤN LUYỆN WORD2VEC BẰNG PYSPARK")
    print("="*80)

    # 1. Initialize Spark Session
    # Sử dụng cấu hình local[4] để chạy 4 luồng trên máy cục bộ
    spark = SparkSession.builder \
        .appName("SparkWord2VecDemo") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    
    print(f"Spark Session đã khởi tạo. Đang tìm dữ liệu tại: {DATA_PATH}")

    try:
        # 2. Load the dataset (Dữ liệu C4 là JSON, mỗi dòng là một object JSON)
        # Sử dụng multiLine=True nếu file JSON là một object lớn, nhưng C4 thường là JSON Lines
        data = spark.read.json(DATA_PATH)
        print(f"Đã tải dữ liệu thành công. Số lượng bản ghi: {data.count()}")
        data.printSchema()

    except Exception as e:
        print(f"\nLỖI TẢI DỮ LIỆU: Vui lòng kiểm tra đường dẫn và sự tồn tại của file JSON.")
        print(f"Đường dẫn đã thử: {DATA_PATH}")
        print(f"Chi tiết lỗi: {e}")
        spark.stop()
        return

    # 3. Preprocessing
    print("\n3. Bắt đầu tiền xử lý và tokenization...")

    # 1. Select the text column and convert to lowercase
    # 2. Remove punctuation, numbers, and special characters (chỉ giữ lại chữ cái và khoảng trắng)
    # 3. Split the text into an array of words (tokens)
    document_df = data.select(lower(col("text")).alias("text_lower"))

    tokenized_df = document_df.withColumn(
        "text_cleaned",
        regexp_replace(col("text_lower"), r"[^a-z\s]", "") # Loại bỏ mọi thứ không phải chữ cái và khoảng trắng
    ).withColumn(
        "tokens",
        split(col("text_cleaned"), "\\s+") # Tách bằng khoảng trắng
    ).select("tokens")
    
    # Lọc bỏ các mảng rỗng sau khi tokenization và làm sạch
    tokenized_df = tokenized_df.filter(col("tokens").isNotNull() & (col("tokens").getItem(0).isNotNull()))
    
    tokenized_df.show(5, truncate=50)


    # 4. Configure and train the Word2Vec model
    print("\n4. Bắt đầu huấn luyện mô hình Word2Vec (Spark MLlib)...")

    # Khởi tạo Word2Vec estimator
    word2vec = Word2Vec(
        vectorSize=100,           # 100-dimensional vectors
        minCount=5,               # Chỉ bao gồm các từ xuất hiện ít nhất 5 lần
        numPartitions=4,          # Phân vùng dữ liệu để tăng tốc độ xử lý
        inputCol="tokens",
        outputCol="word_vectors"
    )

    # Huấn luyện mô hình
    model = word2vec.fit(tokenized_df)
    print("Huấn luyện mô hình Word2Vec hoàn tất.")


    # 5. Demonstrate the model
    print("\n5. Trình diễn kết quả mô hình:")
    
    # Find synonyms for a word (Tìm 5 từ tương đồng nhất với "computer")
    test_word = "computer"
    
    # Kiểm tra xem từ có trong từ vựng không (Spark không có hàm kiểm tra trực tiếp, nhưng có thể thử tìm)
    try:
        # model.findSynonyms trả về một DataFrame với các cột 'word' và 'similarity'
        synonyms = model.findSynonyms(test_word, 5) 
        
        print(f"\nTop 5 từ tương đồng nhất với '{test_word}':")
        synonyms.show(truncate=False)
        
    except Exception as e:
        print(f"Lỗi khi tìm từ tương đồng cho '{test_word}'. Có thể từ này không đủ tần suất (minCount=5).")
        print(f"Chi tiết lỗi: {e}")
        
    print("="*80)
    
    # Stop the Spark session
    spark.stop()

if __name__ == "__main__":
    main()