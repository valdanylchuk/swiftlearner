package saiml.nn.backprop

import com.typesafe.scalalogging.LazyLogging
import org.specs2.mutable.Specification
import saiml.data.{FisherIris, Mnist}


class BackpropClassifierTest extends Specification with LazyLogging {
  "BackpropClassifier" should {
//    "classify the flowers from the Fisher Iris dataset" >> {
//      val (trainingSet, testSet) = FisherIris.trainingAndTestDataFloat()
//
//      // Normalize the input values to speed up learning
//      def normalize(x: Float): Float = (x - 25) / 25
//
//      val classifier = new BackpropClassifier(trainingSet, 3, 5000, normalize)
//
//      val accuracy = (for ((species, params) <- testSet) yield {
//        classifier.predict(params) == species
//      }).count { x: Boolean => x } / testSet.size.toDouble
//
//      accuracy must be_>(0.8)  // 0.96 is typical
//    }

    "classify the handwritten digits from the MNIST dataset" >> {
      val seed = Some(0L)
      // settings for a quick smoke test
      val nHidden = 200
      val nRepeat = 1
      val learnRate = 1.0f
      val expectedAccuracy = 0.85  // got 0.88

      // try these settings for a better result
      // val nHidden = 300
      // val nRepeat = 20
      // val learnRate = 1.0f
      // val expectedAccuracy = 1.0 //0.9  // got 0.94

      val (trainingSet, testSet) = Mnist.trainingAndTestData()

      val start = System.currentTimeMillis

      // Careful normalization is essential with this simple network.
      def scale(x: Float): Float = x / 256.0f // source byte range to (0; 1)
      val mean = trainingSet.flatMap(_._2).take(10000).map(scale).sum / 10000.0f
      def normalize(x: Float): Float = scale(x) - mean // source byte range to ~(-0.5; 0.5)

      val classifier = new BackpropClassifier(trainingSet, nHidden, nRepeat, learnRate, normalize, seed)

      val accuracyData = for ((digit, params) <- testSet) yield {
        val predicted = classifier.predict(params)
        logger.trace(s"Predicted: $predicted; actual: $digit")
        (predicted == digit, predicted, digit)
      }

      val accuracyPerLabel = accuracyData.map(x => (x._3, x._1)).groupBy(_._1)
        .map(Function.tupled { (label, label_match) =>
          (label, {
            val matchCounts = label_match.groupBy(_._2).map(x =>(x._1, x._2.length))
            matchCounts.getOrElse(true, 0).toDouble /
              (matchCounts.getOrElse(true, 0) + matchCounts.getOrElse(false, 0))
          })
        })
      logger.debug(s"accuracyPerLabel: ${accuracyPerLabel.mkString("\n")}")

      val accuracyPerPredicted = accuracyData.map(x => (x._2, x._1)).groupBy(_._1)
        .map(Function.tupled { (label, label_match) =>
          (label, {
            val matchCounts = label_match.groupBy(_._2).map(x =>(x._1, x._2.length))
            matchCounts.getOrElse(true, 0).toDouble /
              (matchCounts.getOrElse(true, 0) + matchCounts.getOrElse(false, 0))
          })
        })
      logger.debug(s"accuracyPerPredicted: ${accuracyPerPredicted.mkString("\n")}")

      val overallAccuracy = accuracyData.count { x: (Boolean, Int, Int) => x._1 } / testSet.size.toDouble

//      val accuracy = (for ((digit, params) <- testSet) yield {
//        classifier.predict(params) == digit
//      }).count { x: Boolean => x } / testSet.size.toDouble

      val time = (System.currentTimeMillis - start) / 1000.0
      logger.debug(s"MNIST backprop time: ${time}s")

      overallAccuracy must be_>(expectedAccuracy)
    }
  }
}