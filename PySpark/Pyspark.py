from pyspark.sql import SparkSession
# Create a Spark session
spark = SparkSession.builder \
    .appName("SalesCustomersQuery") \
    .getOrCreate()

print("Spark session created successfully.")
print(spark)