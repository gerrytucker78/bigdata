val splits = parsedData.randomSplit(Array(0.6,0.4), seed = 11L)
val training = splits(0)
val test = splits(1)
val model = NaiveBayes.train(training, lambda = 1.0, modelType = "multinomial")
val predictionAndLabel = test.map(p=> (model.predict(p.features),p.label))
val accuracy = 1.0 * predictionAndLabel.filter(x=> x._1 == x._2).count() / test.count()
