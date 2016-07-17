package saiml.bayes

import org.specs2.mutable.Specification

import scala.io.Source
import scala.util.Random


/**
  * We are using the classic Fisher Iris flower dataset as an example.
  *
  * Ref: https://en.wikipedia.org/wiki/Iris_flower_data_set
  *      http://www.math.uah.edu/stat/data/Fisher.html
  */
class GaussianNaiveBayesTest extends Specification {
  lazy val irisData = {
    def toDataRow(csv: String): (Int, Seq[Double]) = {
      val args = csv.split(',')
      val itemClass = args.head.toInt
      val parameters = args.tail.map(_.toDouble)
      (itemClass, parameters)
    }
    val dataFileURI = this.getClass.getClassLoader.getResource("FisherIris.csv").toURI
    val csvData = Source.fromFile(dataFileURI).getLines().toSeq.tail  // skip the header
    csvData.map(toDataRow)
  }

  "GaussianNaiveBayes" should {
    "sort the flowers from the Fisher Iris dataset" >> {
      // Split to 2/3 training and 1/3 test data at random
      val (trainingSet, testSet) = Random.shuffle(irisData).splitAt(irisData.size * 2 / 3)

      val classifier = GaussianNaiveBayes.fromTrainingSet(trainingSet)

      val accuracy = (for ((species, params) <- testSet) yield {
        classifier.predict(params) == species
      }).count { x: Boolean => x } / testSet.size.toDouble

      accuracy must be_>(0.8)  // 0.94 is typical
    }
  }
}