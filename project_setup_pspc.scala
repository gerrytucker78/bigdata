import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint

val data = sc.textFile("/got150030/project/p_sp_c_cd.csv")

val parsedData = data.map {line=>
val parts = line.split(',')
LabeledPoint(parts(3).toDouble, Vectors.dense(parts.dropRight(1).map(_.toDouble)))
}


