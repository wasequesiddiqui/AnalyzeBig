#%%
import os

from pyspark.sql.functions import col
from pyspark.sql import SparkSession

# Change working directory to the script's current directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Create a Spark session
spark = SparkSession.builder \
    .appName("BigMart Sales Analysis") \
    .getOrCreate()

# Read the CSV file into a Spark DataFrame
df = spark.read.csv("BigMart Sales.csv", header=True, inferSchema=True)

#%%
df.printSchema()
#%%
# Show the first few
df.show(5)
# %%
df.select('Item_Identifier','Item_Weight','Item_Fat_Content').show()
# %% Aliasing columns
df.select(
    col('Item_Identifier').alias('Item')
    ,col('Item_Weight').alias('Weight')
    ,col('Item_Fat_Content').alias('Fat_Content')
          ).show()
# %% Filtering data scenario 1
df.filter(col('Item_Fat_Content') == 'Regular').show()

# %% Filtering data scenario 2
df.filter((col('Item_Fat_Content') == 'Regular') & (col('Item_Weight') > 10)).show()
# %% Filtering data scenario 3
df.filter(
    (col('Outlet_Size').isNull()) &
    (col('Item_Fat_Content') == 'Regular') &
    (
        (col('Outlet_Location_Type').like('Tier 1')) |
        (col('Outlet_Location_Type').like('Tier 2'))
    )
).show()
# %%
