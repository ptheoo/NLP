import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def run_spark_pipeline_gbt():
    """
    Phân tích cảm xúc bằng mô hình Gradient-Boosted Trees (GBT).
    """

    print("=== 1. Khởi tạo Spark ===")
    spark = SparkSession.builder.appName("SentimentAnalysis_GBT").getOrCreate()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'data', 'sentiments.csv'))
    data_uri = "file:///" + data_path.replace("\\", "/")

    if not os.path.exists(data_path):
        print(f"Không tìm thấy file dữ liệu tại: {data_path}")
        spark.stop()
        return

    df = spark.read.csv(data_uri, header=True, inferSchema=True)
    df = df.dropna(subset=["text", "sentiment"])
    df = df.withColumn("label", (col("sentiment").cast("integer") + lit(1)) / lit(2))
    train, test = df.randomSplit([0.8, 0.2], seed=42)

    print(f"Train: {train.count()} | Test: {test.count()}")

    print("\n=== 2. Pipeline GBT ===")
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=5000)
    idf = IDF(inputCol="raw_features", outputCol="features")

    gbt = GBTClassifier(featuresCol="features", labelCol="label", maxIter=20)

    pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, gbt])
    model = pipeline.fit(train)
    preds = model.transform(test)

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
    accuracy = evaluator.evaluate(preds, {evaluator.metricName: "accuracy"})
    f1 = evaluator.evaluate(preds, {evaluator.metricName: "f1"})

    # Tính precision và recall thủ công
    TP = preds.filter("label = 1.0 AND prediction = 1.0").count()
    FP = preds.filter("label = 0.0 AND prediction = 1.0").count()
    FN = preds.filter("label = 1.0 AND prediction = 0.0").count()
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    print("\nKẾT QUẢ (Gradient-Boosted Trees)")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    spark.stop()


if __name__ == "__main__":
    run_spark_pipeline_gbt()