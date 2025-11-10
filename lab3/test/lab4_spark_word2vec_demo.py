from pyspark.sql import SparkSession
from pyspark.ml.feature import Word2Vec
# Import thêm array_remove để xử lý mảng token rỗng
from pyspark.sql.functions import col, lower, regexp_replace, split, size, array_remove
import os

# --- THIẾT LẬP HADOOP CHO WINDOWS (BẮT BUỘC) ---
# Đảm bảo đường dẫn này TRỎ ĐÚNG đến thư mục C:\hadoop của bạn
os.environ["HADOOP_HOME"] = "C:\\hadoop" 
os.environ["PATH"] += os.pathsep + "C:\\hadoop\\bin" 

def main():
    print("Khởi tạo SparkSession.")
    # Tăng bộ nhớ driver để xử lý tốt hơn tập dữ liệu C4
    spark = SparkSession.builder.appName("Spark Word2Vec Demo") \
        .master("local[*]") \
        .config("spark.driver.memory", "6g") \
        .getOrCreate()

    print("-" * 10)
    print("Đọc dữ liệu")
    # Đường dẫn tương đối đã được xác nhận là đúng: ../data/c4-train.00000-of-01024-30K.json
    df = spark.read.json("../data/c4-train.00000-of-01024-30K.json") 
    df = df.select("text").dropna()
    print(f"Số dòng đọc được: {df.count()}")

    print("-" * 10)
    print("Tiền xử lý văn bản và Tokenization")
    
    # 1. Chuyển text sang chữ thường
    # 2. Loại bỏ tất cả ký tự không phải chữ cái và khoảng trắng, thay thế chúng bằng khoảng trắng.
    document_df = df.withColumn("text_cleaned", lower(regexp_replace("text", "[^a-zA-Z\\s]", " ")))
    
    # 3. Tách văn bản thành mảng từ (tokens).
    words_data = document_df.withColumn("words", split(col("text_cleaned"), "\\s+"))
    
    # 4. Lọc token rỗng: Sử dụng array_remove để xóa các chuỗi rỗng "" khỏi mảng
    words_data = words_data.withColumn(
        "words", 
        array_remove(col("words"), "")
    ).select("words")
    
    # 5. Lọc bỏ các dòng mà mảng token cuối cùng có kích thước bằng 0
    words_data = words_data.filter(size(col("words")) > 0)

    print(f"Số dòng sau khi lọc các dòng trống: {words_data.count()}")
    print("DataFrame sau khi Tokenization:")
    words_data.show(5, truncate=50)

    print("-" * 10)
    print("Huấn luyện mô hình Word2Vec (Skip-gram)")
    word2vec = Word2Vec(
        vectorSize=100,  # Kích thước vector chuẩn cho ngữ nghĩa
        minCount=5,      # Tần suất tối thiểu của từ
        inputCol="words",
        outputCol="vector"
    )
    
    # Bắt đầu quá trình huấn luyện phân tán trên Spark
    model = word2vec.fit(words_data)

    print("\nTìm các từ tương tự 'computer'")
    # Trình diễn mô hình
    synonyms = model.findSynonyms("computer", 10)
    synonyms.show(truncate=False)

    spark.stop()
    print("\nHoàn thành huấn luyện Spark Word2Vec")

if __name__ == "__main__":
    main()