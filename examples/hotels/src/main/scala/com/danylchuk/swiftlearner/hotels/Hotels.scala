package com.danylchuk.swiftlearner.hotels

import com.danylchuk.swiftlearner.nn.backprop.{BackpropClassifier, BackpropNet}


object Hotels extends App {
  val seed = Some(0L)
  val nHidden = 6
  val nRepeat = 123
  val learnRate = 0.3f

  val (trainingSet, testSetIter) = SearchData.trainingAndTestData()
  val testSet = testSetIter.toStream.map(Function.tupled { (label, data) =>
    (label, data.toSeq)
  })

  val start = System.currentTimeMillis

  println("creating the classifier")

  val nClasses = 100

  def predict(learned: BackpropNet, parameters: Seq[Float]): Int = {
    val predicted = learned.calculateOutput(parameters.toArray)
    predicted.indexOf(predicted.max)
  }

  def accuracy = (for ((digit, params) <- testSet) yield {
    predict(learned, params) == digit
  }).count { x: Boolean => x } / testSet.size.toDouble

  val examples: Seq[(Array[Float], Array[Float])] =
    trainingSet map Function.tupled { (classIdx, params) =>
      val targetOneHot = Array.fill[Float](nClasses)(0.0f)
      targetOneHot(classIdx) = 1.0f
      (params.toArray, targetOneHot)
    }

  val nParams = examples.head._1.length

  val nn = BackpropNet.randomNet(nParams, nHidden, nClasses, seed)
  var learned = nn

  for (i <- 1 to nRepeat) {
   learned = learned.learnSeq(examples, learnRate)

    val time = (System.currentTimeMillis - start) / 1000.0
    println(s"Iteration $i / $nRepeat; time so far: ${time}s")  // fast
//    println(s"Iteration $i / $nRepeat; time so far: ${time}s; accuracy: $accuracy")  // verbose
  }
  println(s"accuracy: $accuracy")

}