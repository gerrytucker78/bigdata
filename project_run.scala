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
  }

  Console.println("*** Average Results [" + loopCount + " runs | " + train + " / " + test + "]: " + (results.sum/loopCount))
}

