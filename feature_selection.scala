import org.apache.spark.mllib.feature.ChiSqSelector
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

val data = sc.textFile("/got150030/project/all_feature_sets.csv")

val parsedData = data.map {line=>
  val parts = line.split(',')
  LabeledPoint(parts(6).toDouble, Vectors.dense(parts.dropRight(1).map(_.toDouble)))
}

val selector = new ChiSqSelector(3)

val transformer = selector.fit(parsedData)

val filteredData = parsedData.map {lp =>
  LabeledPoint(lp.label, transformer.transform(lp.features))
}

filteredData.foreach(x=>println(x))
