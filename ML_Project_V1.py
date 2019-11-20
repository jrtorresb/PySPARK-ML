'''
******************************************************
*** Programa para el proyecto Machine Learning ORCs***
*** Realizado por José Roberto Torres Bello        ***
*** Noviembre 2019                                 ***
****************************************************** 

PySpark es la interfaz que da acceso a Spark mediante el lenguaje de programación Python. 
PySpark es una API desarrollada en Python para la programación de spark y para escribir aplicaciones de spark al estilo de Python, 
aunque el modelo de ejecución subyacente es el mismo para todos los lenguajes API.

Google Colab  se basa en el cuaderno Jupyter, que es una herramienta poderosa que aprovecha las funciones de Google Docs.

Dado que se ejecuta en el servidor de Google, no se necesitam instalar nada en el sistema localmente, ya sea un modelo de Spark o de aprendizaje profundo.

Para ejecutar spark en Google Colab, primero se necesitan instalar todas las dependencias en el entorno Google Colab. Como Apache Spark 2.4.4 con hadoop 2.7, Java 8 y Findspark 
para ubicar spark en el sistema. 
La instalación de las herramientas se puede llevar a cabo dentro del cuaderno Jupyter de la Google Colab ejecutando lo siguiente:



!apt-get install openjdk-8-jdk-headless -qq > /dev/null
!wget -q https://www-us.apache.org/dist/spark/spark-2.4.4/spark-2.4.4-bin-hadoop2.7.tgz
!tar xf spark-2.4.4-bin-hadoop2.7.tgz
!pip install -q findspark

!pip install finspark
!pip install pyspark



Ahora que se ha instalado Spark y Java en Google Colab, es hora de establecer la ruta del entorno que nos permita ejecutar PySpark en elentorno Google Colab.

Establece la ubicación de Java y Spark ejecutando el siguiente código:


import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-2.3.2-bin-hadoop2.7"

Se puede ejecutar una sesión local de spark para probar la instalación:

import findspark
findspark.init()
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[*]").getOrCreate()


'''



'''
# REFERENCIAS:

http://sitiobigdata.com/2019/02/09/google-colab-regresion-lineal-pyspark/#
https://cognitus.fr/spark-mllib-tutorial-complete-classification-workflow/
https://creativedata.atlassian.net/wiki/spaces/SAP/pages/83237142/Pyspark+-+Tutorial+based+on+Titanic+Dataset
https://docs.databricks.com/applications/machine-learning/mllib/binary-classification-mllib-pipelines.html#
https://runawayhorse001.github.io/LearningApacheSpark/classification.html

'''



# Si se quiere correr el programa en GOOGLE COLAB
# Variables de entorno ejecutar directamente en un Notebook de python 3 :
# Para descargar la version de spark desde: https://www-us.apache.org/dist/spark/spark-2.4.4/
# Copiar la direccion del enlace del archivo: spark-2.4.4-bin-hadoop2.7.tgz 

'''
# poner lo siguiente en Google Colab:

!apt-get install openjdk-8-jdk-headless -qq > /dev/null
!wget -q https://www-us.apache.org/dist/spark/spark-2.4.4/spark-2.4.4-bin-hadoop2.7.tgz
!tar xf spark-2.4.4-bin-hadoop2.7.tgz
!pip install -q findspark

!pip install findspark
!pip install pyspark


# Variables de entorno deben apuntar a la version de java 8
# a la carpeta que se descargo de spark
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-2.4.4-bin-hadoop2.7"


# Imprime las variables de entorno
print(os.environ)

# Iniciando una sesion de Spark, a partir de la version 2.0
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("mllib").master("local[*]").getOrCreate()

# Imprime la version de Spark
print("Version de SPARK: ", spark.version)

# Imprime la session, version, master, AppName
print(spark)

# crear Spark Context
sc = spark.sparkContext
# Imprime el contexto, version, master, AppName
print(sc)

# Crear Contexto sql:
from pyspark.sql import  SQLContext

# crear Spark Context
# sc = spark.sparkContext

# Imprime el contexto, version, master, AppName
# print(sc)


# Crear Contexto sql:
sqlContext = SQLContext(spark.sparkContext)
sqlContext

'''


'''
# Programa alternativo para crear una SparkSession:


if __name__=="__main__":
    try:
        from pyspark.sql import SparkSession
    except:
        import findspark
        findspark.init()
        from pyspark.sql import SparkSession
    spark=SparkSession.builder.master("local[8]").appName("ejemplo").getOrCreate()

'''


# ----------------------------------------------------------------------COMANDOS BASICOS------------------------------------------------------------- 
'''
# IMPORTACIONES:
# df = spark.read.json('people.json')
# df = spark.read.csv('customer_churn.csv', inferSchema=True, header=True)
# df = spark.read.format("libsvm").load("data.txt")


# df.show() 
# df.describe().show(), df.describe('capital_gain').show()	
# df.printSchema()
# df.columns
# df.select(['age','name']) selecciona columnas

# Simple Rename
# df.withColumnRenamed('age','supernewage').show()

# Crea nueva columna
# df.withColumn('doubleage',df['age']*2).show()

# DROP COLUMN
# df.drop('education_num')
# df=df.drop("education_num")

# REEMPLAZAR COLUMNAS: Replace -1 in df.pdays with 0
# df= df.withColumn("pdays", when(col("pdays") == -1,0).otherwise(col("pdays")))
# df.select(['pdays']).show(2)

..
Drop files to upload them to session storage

# Drop the missing data
​
You can use the .na functions for missing data. The drop command has the following parameters:
​
    df.na.drop(how='any', thresh=None, subset=None)
    
    * param how: 'any' or 'all'.
    
        If 'any', drop a row if it contains any nulls.
        If 'all', drop a row only if all its values are null.
    
    * param thresh: int, default None
    
        If specified, drop rows that have less than `thresh` non-null values.
        This overwrites the `how` parameter.
        
    * param subset: 
        optional list of column names to consider.



# Drop any row that contains missing data
# df.na.drop().show()

# Has to have at least 2 NON-null values
# df.na.drop(thresh=2).show()

# df.na.drop(subset=["Sales"]).show()
# df.na.drop(how='any').show()

# FILL
# df.na.fill('NEW VALUE').show()
# df.na.fill(0).show()

# En la columna NAME
# df.na.fill('No Name',subset=['Name']).show()


# FUNCIONES SQL
# from pyspark.sql.functions import mean
# mean_val = df.select(mean(df['Sales'])).collect()

# Weird nested formatting of Row object!
# mean_sales = mean_val[0][0]

# Llena con la media
# df.na.fill(mean_sales,["Sales"]).show()



# Mean
# df.groupBy("Company").mean().show()

# Count
# df.groupBy("Company").count().show()

# Max
# df.groupBy("Company").max().show()

# Min
# df.groupBy("Company").min().show()

# Sum
# df.groupBy("Company").sum().show()


# from pyspark.sql.functions import countDistinct, avg, stddev, format_number

# df.select(countDistinct("Sales")).show()
# df.select(countDistinct("Sales").alias("Distinct Sales")).show()
# df.select(avg('Sales')).show()
# df.select(stddev("Sales")).show()

# format_number("col_name",decimal places)
# sales_std.select(format_number('std',2)).show()


# OrderBy
# Ascending
# df.orderBy("Sales").show()

# Descending call off the column itself.
# df.orderBy(df["Sales"].desc()).show()




# SI QUIERES CAMBIAR EL TIPO DE DATO:

# Import all from sql.types
from pyspark.sql.types import *

# Write a function to convert the data type of DataFrame columns:


def convertColumn(df, names, newType):
    for name in names: 
        df = df.withColumn(name, df[name].cast(newType))
    return df 


# List of continuous features
CONTI_FEATURES  = ['age', 'fnlwgt','capital_gain', 'education_num', 'capital_loss', 'hours_week']

# Convert the type
df_string = convertColumn(df_string, CONTI_FEATURES, FloatType())

# Check the dataset
df_string.printSchema()



'''
# --------------------------------------------------------LIBRERIAS BASICAS QUE SIEMPRE SIRVEN-----------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import *

# Toma el tiempo de ejecucion
start_time=time()

# -------------------------------------------------------------CARGA Y VISTA DE DATOS --------------------------------------------------------------------------------

# IMPORTACIONES:
# df = spark.read.json('people.json')
# df = spark.read.csv('customer_churn.csv', inferSchema=True, header=True)
# df = spark.read.format("libsvm").load("data.txt")
data = spark.read.csv('titanic3.csv', inferSchema=True, header=True)

# USANDO PANDAS
import pandas as pd
df=pd.DataFrame(data.take(5), columns=dataset.columns)
# Luego lo regresas usando : df_converted = spark.createDataFrame(df)

# TAMBIEN: 
df_dataset = data.limit(5)
df_dataset.toPandas()


# Imprime el nombres de las columnas
data.columns

# Numero de filas
data.count()

# Imprime los tipos de datos
data.printSchema()
data.dtypes

# Muestra las primeras 10 filas completas 
data.show(10, False)


# Puedes crear una vista y usar sql
# data.createOrReplaceTempView("data")
# sex_freq=spark.sql("SELECT Gender, count(*) FROM data GROUP BY Gender ")
# sex_freq.show()
# Tambien usarlo directamente: spark.sql("SELECT * FROM adult WHERE age == 20").show()


# Estadisticas Basicas
data.describe().show()


# PROPORCION DE CLASES
from pyspark.sql.functions import col
sum_y=dataset.select('income').count()
prop_y=dataset.select('income').groupby(dataset.income).count()

prop_y = prop_y \
    .withColumn('prop_y', 
               (col('count')/sum_y)*100 \
                    )
prop_y.show()


# Grafrica de barras con las clases
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(color_codes=True)
responses = data.groupBy('Survived').count().collect()
categories = [i[0] for i in responses]
counts = [i[1] for i in responses]
ind = np.array(range(len(categories)))
width = 0.5
plt.bar(ind, counts, width=width, color='r')
plt.ylabel('counts')
plt.title('label distribution')
plt.xticks(ind + width/10, categories)
plt.show()


# Agrupa y cuenta
data.groupby('education').count().show()
data.groupBy("education").count().sort("count",ascending=True).show()	
data.select('education').distinct().show()

# Ejemplos de filtros que puedes aplicar
data.filter("education_num<13").select(['education_num',"age"]).show()
data.filter( (dataset["education_num"] < 13) & (dataset['age'] > 20) ).show()
data.filter( (dataset["education_num"] < 13) | (dataset['age'] > 20) ).show()
data.filter(dataset["age"] == 20).show()
data.filter( (dataset["education_num"] < 13) | ~(dataset['age'] > 20) ).show()

# PYTHON OBJECT a diccionario
result = dataset.filter(dataset["age"] == 20).collect()
row = result[0]
row.asDict() # convierte a diccionario


# CROSSTAB
data.crosstab('age', 'income').show()	

# ----------------------------------------------------------------VALORES NULOS--------------------------------------------------------------------------------------------------

from pyspark.sql import functions as F
from pyspark.sql.functions import col, sum
from pyspark.sql.types import *
df_null = data.select(*(F.sum(F.col(c).isNull().cast('Double')).alias(c) for c in data.columns))
# si quieres pasarlo a pandas: df_null = data.select(*(F.sum(F.col(c).isNull().cast('Double')).alias(c) for c in data.columns)).toPandas()
df_null.show()


# ------------------------------------------------------------IMPUTANDO DATOS-----------------------------------------------------------------------------------------------------

# Rellenando sin usar imputador:
# 'max','min','sum'
mean_age=data.agg({"Age":"avg"})
mean_age.show()

data=data.fillna({"Age":29.7})
data.show()



# SI QUIERES USAR IMPUTADOR: pero te va a crear una nueva columna AgeImputed

# from pyspark.ml.feature import Imputer
# imputer = Imputer(strategy='mean', inputCols=['Age'], outputCols=['AgeImputed'])
# imputer_model = imputer.fit(data)
# data = imputer_model.transform(data)


# -----------------------------------------------------------------VALORES UNICOS----------------------------------------------------------------------------------------------

# MUY TARDADO NO ES RECOMENDABLE
for col in data.columns:
 col_count = data.select(col).distinct().count()
 print('{0} - Valores unicos: {1}'.format(col, col_count))


# -------------------------------------------------TRANSFORMACION VARIABLES CATEGORICAS Y VECTORIZACION--------------------------------------------------------------------------

# FUNCION PARA TRATAMIENTO DE VARIABLES CATEGORICAS Y NUMERICAS (Las categorias las convierte a numericas y las vectoriza)
# Estados:
# Binarizer ---> String Indexer ---> OneHotEncoder ---> Vector Assembler ---> Estimator ---> Model
def get_dummy(df,categoricalCols,continuousCols,labelCol):

    from pyspark.ml import Pipeline
    from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
    from pyspark.sql.functions import col

    indexers = [ StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c))
                 for c in categoricalCols ]

    # default setting: dropLast=True
    encoders = [ OneHotEncoder(inputCol=indexer.getOutputCol(),
                 outputCol="{0}_encoded".format(indexer.getOutputCol()))
                 for indexer in indexers ]

    assembler = VectorAssembler(inputCols=[encoder.getOutputCol() for encoder in encoders]
                                + continuousCols, outputCol="features")

    pipeline = Pipeline(stages=indexers + encoders + [assembler])

    model=pipeline.fit(df)
    data = model.transform(df)

    data = data.withColumn('label',col(labelCol))

    return data.select('features','label')



# Despues de que se define la funcion se seleccionan las variables categoricas, numericas y la "label column"    

categoricalColumns=['Gender'] # Variables categoricas
numericCols=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare'] # Variables Numericas
labelCol='Survived' # Target

# Son los parametros para la funcion get dummy
data=get_dummy(data, categoricalColumns, numericCols, labelCol)
data.show()


# ----------------------------------------------------------SEPARANDO DATOS EN TRAINING Y TEST--------------------------------------------------------------
# Randomly split data into training and test sets. set seed for reproducibility


trainingData, testData = data.randomSplit([0.7, 0.3], seed=100)


print("Training Dataset Count: " + str(trainingData.count()))
print("Test Dataset Count: " + str(testData.count()))
trainingData.show()


# ---------------------------------------------------------------CREACION DEL MODELOS---------------------------------------------------------------
# Create initial LogisticRegression model


from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10)

# Train model with Training Data
lrModel = lr.fit(trainingData)



# ------------------------------------------------------------------PREDICCIONES------------------------------------------------------------------
# Predict using the test data and evaluate the predictions
# Make predictions on test data using the transform() method.
# LogisticRegression.transform() will only use the 'features' column.

predictions = lrModel.transform(testData)
predictions.show()

# Puedes ver cuantos predijo mal
predictions.groupBy('label','prediction').count().show()

# ----------------------------------------------------------------EVALUACION DEL MODELO----------------------------------------------------------
# We can use BinaryClassificationEvaluator to evaluate our model. 
# We can set the required column names in rawPredictionCol and labelCol Param and the metric in metricName Param.


# Evaluate model
from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator(labelCol='label', rawPredictionCol="rawPrediction",metricName='areaUnderROC')
evaluator.evaluate(predictions)
# Note that the default metric for the BinaryClassificationEvaluator is areaUnderROC
print(lr.explainParams())

# Summary del modelo
trainingSummary = lrModel.summary
trainingSummary.accuracy
trainingSummary.areaUnderROC

# Graficas
roc = trainingSummary.roc.toPandas()
plt.figure()
plt.plot(roc['FPR'],roc['TPR'], label='ROC curve (area = %0.2f)' % trainingSummary.areaUnderROC)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.ylabel('False Positive Rate')
plt.xlabel('True Positive Rate')
plt.title('ROC Curve (Receiver operating characteristic Graph)')
plt.legend(loc="lower right")
plt.show()
print('Training set areaUnderROC: ' + str(trainingSummary.areaUnderROC))



pr = trainingSummary.pr.toPandas()
plt.plot(pr['recall'],pr['precision'])
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.show()


# Matriz de confusion y classification report usando scikit-learn
# If you want to generate other evaluations such as a confusion matrix or a classification report, you could always use the scikit-learn library.
# You only need to extract y_true and y_pred from your DataFrame. 
y_true = predictions.select(['label']).collect()
y_pred = predictions.select(['prediction']).collect()

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))


# ---------------------------------------------------------------ESTIMACION DE PARAMETROS---------------------------------------------------------
# Estimacion de los parametros y validacion cruzada


from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Create ParamGrid for Cross Validation
paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.01, 0.5, 2.0])
             .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
             .addGrid(lr.maxIter, [1, 5, 10])
             .build())

# Create 5-fold CrossValidator
cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)

# Run cross validations
cvModel = cv.fit(trainingData)
# this will likely take a fair amount of time because of the amount of models that we're creating and testing           

# Mejores parametros
cvModel.bestModel.extractParamMap()  

# Use test set to measure the accuracy of our model on new data
predictions = cvModel.transform(testData)

# cvModel uses the best model found from the Cross Validation
# Evaluate best model
evaluator.evaluate(predictions)

print("Done!")

# ----------------------------------------------------------------TIEMPO TOTAL DE EJECUCION----------------------------------------------------------------------------------------

end_time=time()

time_in_minutes = int(float(end_time-start_time)/60)

print("Tiempo de ejecuccion: ", time_in_minutes, " minutos")