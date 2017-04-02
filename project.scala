import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.util.MLUtils
import org.apache.log4j.Logger
import org.apache.log4j.Level

Logger.getLogger("org").setLevel(Level.OFF)
Logger.getLogger("akka").setLevel(Level.OFF)


def calcAccuracy(parsedData:RDD[LabeledPoint], loopCount:Int, train:Double, test:Double)={
  val split = Array(train,test)
  var results : List[Double] = List() 
  
  /** Naive Baysean **/
  for (i <- 1 to loopCount) yield {
    val seed = System.currentTimeMillis()
    val splits = parsedData.randomSplit(split, seed)
    val training = splits(0)
    val test = splits(1)
    val model = NaiveBayes.train(training, lambda = 1.0, modelType = "multinomial")
    val predictionAndLabel = test.map(p=> (model.predict(p.features),p.label))
    val accuracy :Double = 1.0 * predictionAndLabel.filter(x=> x._1 == x._2).count() / test.count()
    val result = accuracy
    results :::= List(accuracy)
  }

  Console.println("*** NaiveBayes Average Results [" + loopCount + " runs | " + train + " / " + test + "]: " + (results.sum/loopCount))


  /** SVM Linear Regression **/

  var svmResults : List[Double] = List() 
  for (i <- 1 to 1) yield {
    val seed = System.currentTimeMillis()
    val splits = parsedData.randomSplit(split, seed)
    val training = splits(0)
    val test = splits(1)
  
    val svmModel = SVMWithSGD.train(training,loopCount);
    svmModel.clearThreshold()

    // Compute raw scores on the test set.
    val scoreAndLabels = test.map { point =>
    val score = svmModel.predict(point.features)
      (score, point.label)
    }

    // Get evaluation metrics.
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val accuracy = metrics.areaUnderROC()
    val result = accuracy
    svmResults :::= List(accuracy)
  }

  Console.println("*** Linear SVM Average Results [" + loopCount + " runs | " + train + " / " + test + "]: " + (svmResults.sum))

  /** Decision Tree **/
  var dtResults : List[Double] = List() 
  for (i <- 1 to loopCount) {
    val seed = System.currentTimeMillis()
    val splits = parsedData.randomSplit(split, seed)
    val training = splits(0)
    val test = splits(1)

    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val impurity = "gini"
    val maxDepth = 5
    val maxBins = 32

    val model = DecisionTree.trainClassifier(training, numClasses, categoricalFeaturesInfo,
      impurity, maxDepth, maxBins)

    // Create tuples of predicted and real labels.
    val predictionAndLabel = test.map { point =>
      val predictedLabel = model.predict(point.features)
      (predictedLabel, point.label)
    }

    val accuracy :Double = 1.0 * predictionAndLabel.filter(x=> x._1 == x._2).count() / test.count()

    // Calculate mean squared error between predicted and real labels.
    //val meanSquaredError = predictionAndLabel.map { case (p, l) => math.pow((p - l), 2) }.mean()
    dtResults :::= List(accuracy)
  }
  Console.println("*** Decision Tree Average Results [" + loopCount + " runs | " + train + " / " + test + "]: " + (dtResults.sum / loopCount))
}

/*** Product / Sub-Product / Company **/
val data = sc.textFile("/got150030/project/p_sp_c_cd.csv")

val parsedData = data.map {line=>
  val parts = line.split(',')
  LabeledPoint(parts(3).toDouble, Vectors.dense(parts.dropRight(1).map(_.toDouble)))
}

Console.println("\n\n********** Product, Sub-Product, Company **********")
calcAccuracy(parsedData,50,.7,.3)
calcAccuracy(parsedData,50,.8,.2)
calcAccuracy(parsedData,50,.9,.1)


/*** Product / Sub-Product / Issue / Company **/
val data = sc.textFile("/got150030/project/p_sp_i_c_cd.csv")

val parsedData = data.map {line=>
  val parts = line.split(',')
  LabeledPoint(parts(4).toDouble, Vectors.dense(parts.dropRight(1).map(_.toDouble)))
}

Console.println("\n\n********** Product, Sub-Product, Issue, Company **********")
calcAccuracy(parsedData,50,.7,.3)
calcAccuracy(parsedData,50,.8,.2)
calcAccuracy(parsedData,50,.9,.1)

/*** Product / Sub-Product / Company / State **/
val data = sc.textFile("/got150030/project/p_sp_c_s_cd.csv")

val parsedData = data.map {line=>
  val parts = line.split(',')
  LabeledPoint(parts(4).toDouble, Vectors.dense(parts.dropRight(1).map(_.toDouble)))
}

Console.println("\n\n********** Product, Sub-Product, Company, State **********")
calcAccuracy(parsedData,50,.7,.3)
calcAccuracy(parsedData,50,.8,.2)
calcAccuracy(parsedData,50,.9,.1)

/*** Product / Sub-Product / Company / Year **/
val data = sc.textFile("/got150030/project/p_sp_c_y_cd.csv")

val parsedData = data.map {line=>
  val parts = line.split(',')
  LabeledPoint(parts(4).toDouble, Vectors.dense(parts.dropRight(1).map(_.toDouble)))
}

Console.println("\n\n********** Product, Sub-Product, Company, Year **********")
calcAccuracy(parsedData,50,.7,.3)
calcAccuracy(parsedData,50,.8,.2)
calcAccuracy(parsedData,50,.9,.1)

/*** Product / Company / State **/
val data = sc.textFile("/got150030/project/p_c_s_cd.csv")

val parsedData = data.map {line=>
  val parts = line.split(',')
  LabeledPoint(parts(3).toDouble, Vectors.dense(parts.dropRight(1).map(_.toDouble)))
}

Console.println("\n\n********** Product, Company, State **********")
calcAccuracy(parsedData,50,.7,.3)
calcAccuracy(parsedData,50,.8,.2)
calcAccuracy(parsedData,50,.9,.1)

/*** Sub-Product / Issue / Company **/
val data = sc.textFile("/got150030/project/sp_i_c_cd.csv")

val parsedData = data.map {line=>
  val parts = line.split(',')
  LabeledPoint(parts(3).toDouble, Vectors.dense(parts.dropRight(1).map(_.toDouble)))
}

Console.println("\n\n********** Sub-Product, Issue, Company **********")
calcAccuracy(parsedData,50,.7,.3)
calcAccuracy(parsedData,50,.8,.2)
calcAccuracy(parsedData,50,.9,.1)

/*** Sub-Product / Company **/
val data = sc.textFile("/got150030/project/sp_c_cd.csv")

val parsedData = data.map {line=>
  val parts = line.split(',')
  LabeledPoint(parts(2).toDouble, Vectors.dense(parts.dropRight(1).map(_.toDouble)))
}

Console.println("\n\n********** Sub-Product, Company **********")
calcAccuracy(parsedData,50,.7,.3)
calcAccuracy(parsedData,50,.8,.2)
calcAccuracy(parsedData,50,.9,.1)

