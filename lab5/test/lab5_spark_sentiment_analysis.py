import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def run_spark_pipeline():
    """
    Khởi tạo Spark, tải dữ liệu, huấn luyện pipeline và đánh giá mô hình.
    """
    
    # --- 1. Khởi tạo Spark Session ---
    print("--- 1. Khởi tạo Spark Session ---")
    try:
        spark = SparkSession.builder.appName("SentimentAnalysis").getOrCreate()
        print("Spark Session đã được khởi tạo thành công.")
    except Exception as e:
        print(f"LỖI: Không thể khởi tạo Spark Session. Đảm bảo PySpark đã được cài đặt. Chi tiết: {e}")
        return

    # Xác định đường dẫn tương đối đến file sentiments.csv
   
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'data', 'sentiments.csv'))
    
    # Chuyển đổi đường dẫn thành định dạng URL cho Spark
    
    sanitized_path = data_path.replace('\\', '/')
    data_path_uri = f"file:///{sanitized_path}"

    if not os.path.exists(data_path):
        print(f"\nLỖI: Không tìm thấy file dữ liệu tại đường dẫn: {data_path}")
        print("Vui lòng đảm bảo file sentiments.csv nằm trong thư mục NLP/data/")
        spark.stop()
        return

    # --- 2. Tải và Tiền xử lý Dữ liệu ---
    print("\n--- 2. Tải và Tiền xử lý Dữ liệu ---")
    
    # Tải dữ liệu
    df = spark.read.csv(data_path_uri, header=True, inferSchema=True)

    # Chuyển đổi nhãn từ -1/1 sang 0/1 (0 cho Negative, 1 cho Positive)
    # Công thức: (sentiment + 1) / 2
    df = df.withColumn("label", (col("sentiment").cast("integer") + lit(1)) / lit(2))
    
    # Đếm và làm sạch dữ liệu
    initial_row_count = df.count()
    df = df.dropna(subset=["sentiment", "text"])
    cleaned_row_count = df.count()
    
    # Chia dữ liệu thành tập huấn luyện (80%) và kiểm tra (20%)
    (trainingData, testData) = df.randomSplit([0.8, 0.2], seed=42)

    print(f"Số hàng ban đầu: {initial_row_count}, Số hàng sau khi làm sạch: {cleaned_row_count}")
    print(f"Dữ liệu huấn luyện: {trainingData.count()} hàng. Dữ liệu kiểm tra: {testData.count()} hàng.")

    # --- 3. Xây dựng Pipeline Xử lý Văn bản ---
    print("\n--- 3. Xây dựng Pipeline Xử lý Văn bản ---")
    
    # a. Tokenizer: Tách văn bản thành từ
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    
    # b. StopWordsRemover: Loại bỏ từ dừng tiếng Anh
    stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    
    # c. HashingTF: Chuyển token thành vector đặc trưng thưa (sparse vector)
    HASHTF_FEATURES = 10000
    hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=HASHTF_FEATURES)
    
    # d. IDF: Tính trọng số IDF
    idf = IDF(inputCol="raw_features", outputCol="features")

    # --- 4. Huấn luyện Mô hình Logistic Regression ---
    print("\n--- 4. Huấn luyện Mô hình Logistic Regression ---")
    
    # e. LogisticRegression: Mô hình phân loại nhị phân
    lr = LogisticRegression(maxIter=10, regParam=0.001, featuresCol="features", labelCol="label")
    
    # f. Lắp ráp Pipeline: Các bước tiền xử lý và mô hình
    pipeline = Pipeline(stages=[tokenizer, stopwordsRemover, hashingTF, idf, lr])
    
    model = pipeline.fit(trainingData)
    print("Huấn luyện Pipeline hoàn tất.")

    # --- 5. Đánh giá Mô hình (Tính toán đầy đủ 4 Metrics) ---
    print("\n--- 5. Đánh giá Mô hình ---")
    
    # Dự đoán trên tập kiểm tra
    predictions = model.transform(testData)
    
    # Hiển thị 5 mẫu dự đoán đầu tiên để kiểm tra
    print("5 Mẫu dự đoán đầu tiên (Nhãn thực tế vs Dự đoán):")
    predictions.select("label", "prediction", "text").show(5, truncate=30) 
    
    # 5a. Tính toán Accuracy và F1-Score (Sử dụng MulticlassClassificationEvaluator)
    mc_evaluator = MulticlassClassificationEvaluator(
        labelCol="label", 
        predictionCol="prediction"
    )

    accuracy = mc_evaluator.evaluate(predictions, {mc_evaluator.metricName: "accuracy"})
    f1_score_val = mc_evaluator.evaluate(predictions, {mc_evaluator.metricName: "f1"})

    # 5b. Tính toán Precision và Recall
    
    # TP: True Positive (label=1, prediction=1)
    TP = predictions.filter("label = 1.0 AND prediction = 1.0").count()
    # FP: False Positive (label=0, prediction=1)
    FP = predictions.filter("label = 0.0 AND prediction = 1.0").count()
    # FN: False Negative (label=1, prediction=0)
    FN = predictions.filter("label = 1.0 AND prediction = 0.0").count()
    
    # Tính toán Precision và Recall
    precision_val = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall_val = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    # In kết quả đánh giá cuối cùng (Đã thêm Precision và Recall)
    print("\n KẾT QUẢ ĐÁNH GIÁ CUỐI CÙNG (PySpark) ")
    print("-" * 45)
    print(f"- Accuracy: {accuracy:.4f}")
    print(f"- Precision: {precision_val:.4f}")
    print(f"- Recall: {recall_val:.4f}")
    print(f"- F1_score: {f1_score_val:.4f}")
    print("-" * 45)

    # Dừng Spark Session
    spark.stop()

if __name__ == "__main__":
    run_spark_pipeline()