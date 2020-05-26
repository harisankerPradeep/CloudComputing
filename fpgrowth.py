import findspark
findspark.init()
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml.fpm import FPGrowth
sc = SparkContext('local')
spark = SparkSession(sc)

txt = sc.textFile("./output/*/*")
temp_var = txt.map(lambda k:(0, list(set(k.split(" ")))))
df = temp_var.toDF(["id","words"])

fpGrowth = FPGrowth(itemsCol="words", minSupport=0.1, minConfidence=0.1)
model = fpGrowth.fit(df)

# Display frequent itemsets.
model.freqItemsets.show()

# Display generated association rules.
model.associationRules.show()

# transform examines the input items against all the association rules and summarize the
# consequents as prediction
model.transform(df).show()
