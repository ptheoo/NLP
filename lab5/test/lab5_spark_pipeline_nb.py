# NLP/lab5/test/spark_sentiment_nb.py
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def run_spark_pipeline_nb():
    """
    Phân tích cảm xúc sử dụng PySpark ML Pipeline với Naive Bayes.
    Đánh giá bằng Accuracy, Precision, Recall, F1-score.
    """

    # --- 1. Khởi tạo Spark ---
    print("=== 1. Khởi tạo Spark Session ===")
    spark = SparkSession.builder.appName("SentimentAnalysis_NaiveBayes").getOrCreate()
    print("Spark Session khởi tạo thành công.")

    # --- 2. Đọc dữ liệu ---
    print("\n=== 2. Tải và xử lý dữ liệu ===")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'data', 'sentiments.csv'))
    data_path_uri = "file:///" + data_path.replace("\\", "/")

    if not os.path.exists(data_path):
        print(f"Không tìm thấy file dữ liệu tại: {data_path}")
        spark.stop()
        return

    # Đọc CSV
    df = spark.read.csv(data_path_uri, header=True, inferSchema=True)

    # Chuyển sentiment (-1, 1) -> label (0, 1)
    df = df.withColumn("label", (col("sentiment").cast("integer") + lit(1)) / lit(2))

    # Loại bỏ dòng null
    df = df.dropna(subset=["sentiment", "text"])
    total = df.count()
    print(f"Số dòng dữ liệu sau khi làm sạch: {total}")

    # Chia tập huấn luyện / kiểm thử
    train, test = df.randomSplit([0.8, 0.2], seed=42)
    print(f"Train: {train.count()} | 🧪 Test: {test.count()}")

    # --- 3. Xây dựng pipeline ---
    print("\n=== 3. Xây dựng pipeline Naive Bayes ===")
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=5000)
    idf = IDF(inputCol="raw_features", outputCol="features")
    nb = NaiveBayes(featuresCol="features", labelCol="label", smoothing=1.0, modelType="multinomial")

    pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, nb])

    # --- 4. Huấn luyện ---
    print("\n=== 4. Huấn luyện mô hình ===")
    model = pipeline.fit(train)
    print("Mô hình Naive Bayes huấn luyện xong.")

    # --- 5. Dự đoán & đánh giá ---
    print("\n=== 5. Đánh giá mô hình ===")
    preds = model.transform(test)

    preds.select("label", "prediction", "text").show(5, truncate=40)

    # 5a. Accuracy & F1-score
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
    accuracy = evaluator.evaluate(preds, {evaluator.metricName: "accuracy"})
    f1_score = evaluator.evaluate(preds, {evaluator.metricName: "f1"})

    # 5b. Precision & Recall (tự tính)
    TP = preds.filter("label = 1.0 AND prediction = 1.0").count()
    FP = preds.filter("label = 0.0 AND prediction = 1.0").count()
    FN = preds.filter("label = 1.0 AND prediction = 0.0").count()

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    # --- 6. In kết quả ---
    print("\nKẾT QUẢ ĐÁNH GIÁ MÔ HÌNH (Naive Bayes)")
    print("-" * 50)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1_score:.4f}")
    print("-" * 50)

    # --- 7. Dừng Spark ---
    spark.stop()
    print("\nĐã dừng Spark Session.")


if __name__ == "__main__":
    run_spark_pipeline_nb()
