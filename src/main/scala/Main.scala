package edu.neu.coe.csye7200

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.{DataFrame, Row, SaveMode, SparkSession}
import org.apache.spark.sql.functions._

object Main extends App {
  val trainSrc = "src/in/train.csv";
  val testSrc = "src/in/test.csv";

  val spark: SparkSession = SparkSession.builder().appName("Titanic Survival").master("local[1]").getOrCreate();
  def getDataFrame(fileSrc: String, conf: Map[String, String] = Map("delimiter" -> ",", "header" -> "true", "inferSchema" -> "true")): DataFrame = spark.read.options(conf).csv(fileSrc).toDF();

  val trainDf = getDataFrame(trainSrc);
  val testDf = getDataFrame(testSrc);

  // check for NUll Data Analysis
  println(trainDf.where("Age is null").count());
  println(trainDf.where("Pclass is null").count());
  println(trainDf.where("Sex is null").count());

  trainDf.groupBy("Pclass", "Survived").count().as("Count").show();
  trainDf.groupBy("Sex", "Survived").count().as("Count").show();

  println("Train Data")
  trainDf.describe("Pclass", "Sex", "Age").show();

  val survivors = trainDf.filter(col("Survived") === "1")

  // Create a new column "AgeGroup" to represent age groups of 10
  val survivorsWithAgeGroup = survivors.withColumn("Age_Group", floor(col("Age") / 10) * 10)

  // Group by the "AgeGroup" column and count the number of survivors in each group
  val survivorsByAge = survivorsWithAgeGroup.groupBy("Age_Group").agg(count("*").alias("Survivors_Count"))

  // Show the results
  survivorsByAge.orderBy("Age_Group").show()

  val avgAge = trainDf.select("Age").agg(avg("Age")).collect() match {
    case Array(Row(avg: Double)) => avg
    case _ => 0
  };

  val fillDf = trainDf.na.fill(Map("Age" -> avgAge));
  val Array(toTrainDf,accuracyTestDf) = fillDf.randomSplit(Array(0.7, 0.3));
  // Index Categorical Data like (Sex, Pclass)
  val pClassIndexer = new StringIndexer()
    .setInputCol("Pclass")
    .setOutputCol("PClass_Indexed").fit(fillDf);

  val sexIndexer = new StringIndexer()
    .setInputCol("Sex")
    .setOutputCol("Sex_Indexed").fit(fillDf);

  val labelIndexer = new StringIndexer()
    .setInputCol("Survived")
    .setOutputCol("Survived_Indexed").fit(fillDf);

  // Numbered and Indexed Categorical columns , No Survived Column in these Features
  val allFeatureColNames = Seq("Age", "SibSp", "Parch", "PClass_Indexed", "Sex_Indexed");
  val assembler = new VectorAssembler()
    .setInputCols(Array(allFeatureColNames: _*))
    .setOutputCol("Features")

  val randomForest = new RandomForestClassifier()
    .setLabelCol("Survived_Indexed")
    .setFeaturesCol("Features");

  //Retrieve original labels
  val labelConverter = new IndexToString()
    .setInputCol("prediction")
    .setOutputCol("predicted_Label")
    .setLabels(labelIndexer.labelsArray.flatten);

  val classifierPipeLine =  new Pipeline()
    .setStages((Seq(pClassIndexer, sexIndexer) :+ labelIndexer :+ assembler :+ randomForest :+ labelConverter).toArray);

  // grid of values to perform cross validation on
  val paramGrid = new ParamGridBuilder()
    .addGrid(randomForest.maxBins, Array(25, 28, 31))
    .addGrid(randomForest.maxDepth, Array(4, 6, 8))
    .addGrid(randomForest.impurity, Array("entropy", "gini"))
    .build()

  val evaluator = new BinaryClassificationEvaluator()
    .setLabelCol("Survived_Indexed")

  val cv = new CrossValidator()
    .setEstimator(classifierPipeLine)
    .setEvaluator(evaluator)
    .setEstimatorParamMaps(paramGrid)
    .setNumFolds(10)

  // train the model
  val crossValidatorModel = cv.fit(toTrainDf)

  def prepareTestData(df: DataFrame): DataFrame = {
    val avgAge = df.select("Age").agg(avg("Age")).collect() match {
      case Array(Row(avg: Double)) => avg
      case _ => 0
    };
    df.na.fill(Map("Age" -> avgAge));
  }

  // AUC-ROC
  //Accuracy against split trained data
  val predictions = crossValidatorModel.transform(accuracyTestDf);
  val accuracy = evaluator.evaluate(predictions);
  println(s"Test Error: ${(1.0 - accuracy)*100}%");
  println(s"Accuracy: ${accuracy*100}%");

  // Predict test.csv in predicted_Label
  // write predicted file to src/out/
 val testPrediction = crossValidatorModel.transform(prepareTestData(testDf));
 testPrediction.drop("Features","probability","prediction","rawPrediction").write.mode(SaveMode.Overwrite).options(Map("header" -> "true")).csv("src/out/predictTest.csv");
}