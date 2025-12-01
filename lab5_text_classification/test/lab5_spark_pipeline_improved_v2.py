import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, regexp_replace, lower
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def run_spark_pipeline_improved_v2():
    """
    Phiên bản cải tiến của Logistic Regression:
    - Làm sạch văn bản (xóa ký tự đặc biệt, link, số)
    - Tối ưu số chiều TF-IDF
    - Đánh giá Accuracy, Precision, Recall, F1-score
    """

    print("=== 1. Khởi tạo Spark ===")
    spark = SparkSession.builder.appName("SentimentAnalysis_ImprovedV2").getOrCreate()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'data', 'sentiments.csv'))
    data_uri = "file:///" + data_path.replace("\\", "/")

    if not os.path.exists(data_path):
        print(f"Không tìm thấy file dữ liệu tại: {data_path}")
        spark.stop()
        return

    print("\n=== 2. Tải & Làm sạch dữ liệu ===")
    df = spark.read.csv(data_uri, header=True, inferSchema=True)
    df = df.dropna(subset=["text", "sentiment"])
    df = df.withColumn("label", (col("sentiment").cast("integer") + lit(1)) / lit(2))

    # Làm sạch văn bản
    df = df.withColumn("text", lower(col("text")))
    df = df.withColumn("text", regexp_replace(col("text"), r"http\S+|www\S+", ""))
    df = df.withColumn("text", regexp_replace(col("text"), r"[^a-zA-Z\s]", ""))
    df = df.withColumn("text", regexp_replace(col("text"), r"\s+", " "))

    train, test = df.randomSplit([0.8, 0.2], seed=42)

    print(f"Train: {train.count()} | Test: {test.count()}")

    print("\n=== 3. Pipeline cải tiến ===")
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=8000)
    idf = IDF(inputCol="raw_features", outputCol="features")
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=20, regParam=0.01)

    pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, lr])

    model = pipeline.fit(train)
    preds = model.transform(test)

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
    accuracy = evaluator.evaluate(preds, {evaluator.metricName: "accuracy"})
    f1 = evaluator.evaluate(preds, {evaluator.metricName: "f1"})

    TP = preds.filter("label = 1.0 AND prediction = 1.0").count()
    FP = preds.filter("label = 0.0 AND prediction = 1.0").count()
    FN = preds.filter("label = 1.0 AND prediction = 0.0").count()
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    print("\nKẾT QUẢ CẢI TIẾN (Logistic Regression)")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    spark.stop()


if __name__ == "__main__":
    run_spark_pipeline_improved_v2()
