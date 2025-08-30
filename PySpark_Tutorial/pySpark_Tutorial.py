#%%
import os
import numpy as np
import pandas as pd

from pyspark.sql.functions import *
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

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
#%% sorting data
df.sort(col('Item_Identifier').asc(), col('Item_Weight').desc()).show(5)

#%% Initicap function upper() and lower()
df.select(initcap(col('Item_Type')).alias('Type_CamelCase')).distinct().show()

# %%
df.select(upper(col('Item_Type')).alias('Type_UpperCase')).distinct().show()

# %%
df.select(lower(col('Item_Type')).alias('Type_LowerCase')).distinct().show()

# %% use of current date function
df = df.withColumn('Current_Sys_Date',current_date())
df.show()
# %% date add function adding 7 days
df = df.withColumn('Adding 1 week lates',date_add('Current_Sys_Date',7))
df = df.withColumnRenamed("Adding 1 week lates", "Future_week_Date")
df.show()
# %% date subtract 
df = df.withColumn("Prior_Week_Date", date_add('Current_Sys_Date',-8))
df.show()
# %%
df = df.withColumn("Prior_Week_Date", date_diff('Future_week_Date','Prior_Week_Date'))
df.show()
# %% fillna
df = df.na.fill({'Item_Weight':0,'Outlet_Size':'Unknown'})
df.show()
# %% dropna
df = df.na.drop(subset=['Item_Weight','Outlet_Size'])
df.show()
# %% split function
df = df.withColumn('Outlet_Type_Split', split(col('Outlet_Type'), ' '))
df.show()
# %% explode function
df_exploded = df.select('Outlet_Type', explode(col('Outlet_Type_Split')).alias('Outlet_Type_Exploded'))
df_exploded.show()

# %% array_contains
df = df.withColumn('Contains_Supermarket', array_contains(col('Outlet_Type_Split'), 'Supermarket'))
df.show()
#%% groupby and aggregation
df_grouped = df.groupBy('Item_Type').agg(round(avg('Item_Weight'),2).alias('Avg_Item_Weight'),
                                                 max('Item_Weight').alias('Max_Item_Weight'),
                                                    min('Item_Weight').alias('Min_Item_Weight'),
                                                    count('Item_Weight').alias('Count_Item_Weight'),
                                                    round(sum('Item_Mrp'),2).alias('Sum_Item_MRP'),
                                                    round(avg('Item_MRP'),2).alias('Avg_Item_MRP')
                                                 ).orderBy('Avg_Item_MRP', ascending=False)
df_grouped.show()
# %% collect list
df_grouped_list = df.groupBy('Item_Type').agg(collect_list('Item_Fat_Content').alias('Item_Fat_Content_List'))
df_grouped_list.show(truncate=False)

# %%collect list
df_grouped_set = df.groupBy('Item_Type').agg(collect_set('Item_Fat_Content').alias('Item_Fat_Content_List'))
df_grouped_set.show(truncate=False)
# %% pivot function
df_pivot = df.groupBy('Item_Type').pivot('Item_Fat_Content').agg(count('Item_Fat_Content').alias('Count'),
                                                                 round(avg('Item_Weight'),2).alias('Avg_Weight'),
                                                                 round(sum('Item_MRP'),2).alias('Sum_MRP')
                                                                 ).orderBy('Item_Type')
df_pivot.show(truncate=False)
# %%
