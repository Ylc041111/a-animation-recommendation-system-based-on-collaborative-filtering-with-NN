from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import MinHashLSH
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.sql import functions as F
from pyspark.ml.feature import MinHashLSHModel
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
import csv
conf = SparkConf().setAppName("UserRecommendation").setMaster("local[*]") # 或者其他适用的master URL
conf.set("spark.default.parallelism", "2000") # 设置并行度为200
#创建Spark会话
spark = SparkSession.builder.config(conf=conf).getOrCreate()
#设置分区数量
num_partitions = 4 

#读取原始数据集并设置分区
data = spark.read.csv('/home/user/user-filtered.csv', header=True, inferSchema=True).repartition(num_partitions)
print(data)
data = data.withColumn('user_id', col('user_id').cast(IntegerType()))
#合并原始数据与新用户数据
combined_data = data
#将用户-动漫关联矩阵转换为特征向量
assembler = VectorAssembler(inputCols=combined_data.columns[1:], outputCol="features")
output = assembler.transform(combined_data)
#创建标准化器
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withMean=True, withStd=True)
scalerModel = scaler.fit(output)
output = scalerModel.transform(output)
#保存预处理后的数据
output.write.parquet("/home/user/data/preprocessed_data.parquet", mode="overwrite")
print("ok")
#训练MinHashLSH模型
mh = MinHashLSH(inputCol="scaledFeatures", outputCol="hashes", numHashTables=5)
model = mh.fit(output)
model_path = "/home/user/minhash_lsh_model"

#model = MinHashLSHModel.load(model_path)

def get_user_recommendations(user_id, num_recommendations=5):
    user_features = output.filter(col("user_id") == user_id).select("scaledFeatures").collect()[0]["scaledFeatures"]
    user_feature_vector = Vectors.dense(user_features.toArray())
    similar_users = model.approxNearestNeighbors(output, user_feature_vector, numNearestNeighbors=num_recommendations)

    recommendations = similar_users.select("user_id").rdd.flatMap(lambda x: x).collect()

    return recommendations

#model.save('/home/user/minhash_lsh_model') 
#print("ok")
user_id = 35
recommendations = get_user_recommendations(user_id, num_recommendations=5)
print("推荐给用户{}的动漫：".format(user_id))
for j, recommended_user_id in enumerate(recommendations):
    print("推荐用户 {}: {}".format(j + 1, recommended_user_id))

spark.stop()