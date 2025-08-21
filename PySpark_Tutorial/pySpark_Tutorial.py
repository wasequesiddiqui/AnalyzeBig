#%%
import os
from xml.dom.minicompat import StringTypes
import numpy as np
import pandas as pd

from pyspark.sql.functions import col,lit,round,regexp_replace,concat
from pyspark.sql import SparkSession
# from pyspark.sql.types import StringType

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
# %% withcolumn Function
df= df.withColumn('Flag',lit("New"))
df.show(5)
# %% withcolumn transformation and round result to 4 digits
df = df.withColumn('MRP_Per_Kg', round(col('Item_MRP') / col('Item_Weight'), 4))
df.show(5)
# %%withcolumn transformation regexp replace
df = df.withColumn('Item_Fat_Content',
    regexp_replace(col('Item_Fat_Content'), 'Reg', 'REG')
    ).withColumn('Item_Fat_Content',
    regexp_replace(col('Item_Fat_Content'), 'Low', 'LOW'))
df.show(5)
# %% Casting data types
df = df.withColumn('Outlet_Establishment_Year', col('Outlet_Establishment_Year').cast(StringType()))
df.printSchema()
# %% Adding a new column with a condition
df = df.withColumn('Item_visibility_Chances'
                   , round(col('Item_Visibility') * 100, 2)).withColumn(
                'Item_visibility_Chances', col('Item_visibility_Chances').cast(StringType())).withColumn(
                'Item_visibility_Chances', concat(
                 col('Item_visibility_Chances'), lit(' %')))
df.show()
# %% sorting data
df.sort(col('Item_Identifier').asc(), col('Item_Weight').desc()).show(5)

# %%