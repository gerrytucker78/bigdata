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
LabeledPoint(parts(3).toDouble, Vectors.dense(parts.dropRight(1).map(_.toDouble)))
}


calcAccuracy(50,.7,.3)
calcAccuracy(50,.8,.2)
calcAccuracy(50,.9,.1)

