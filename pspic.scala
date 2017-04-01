import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint

import org.apache.log4j.Logger
import org.apache.log4j.Level

Logger.getLogger("org").setLevel(Level.OFF)
Logger.getLogger("akka").setLevel(Level.OFF)

val data = sc.textFile("/got150030/project/p_sp_i_c_cd.csv")

val parsedData = data.map {line=>
val parts = line.split(',')
LabeledPoint(parts(4).toDouble, Vectors.dense(parts.dropRight(1).map(_.toDouble)))
}

def calcAccuracy(loopCount:Int, train:Double, test:Double)={
  val split = Array(train,test)
  var results : List[Double] = List() 
  
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
    Console.print(i + "..");
  }

  Console.println("\n*** Average Results [" + loopCount + " runs | " + train + " / " + test + "]: " + (results.sum/loopCount))
}


calcAccuracy(50,.7,.3)
calcAccuracy(50,.8,.2)
calcAccuracy(50,.9,.1)

