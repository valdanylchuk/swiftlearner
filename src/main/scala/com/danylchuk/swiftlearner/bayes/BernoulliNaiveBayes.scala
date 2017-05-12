package com.danylchuk.swiftlearner.bayes

import com.danylchuk.swiftlearner.math.Stat

import scala.annotation.tailrec
import scala.collection.mutable


/**
  * Bernoulli naive Bayes classifier.
  *
  * You can create it first time with fromTrainingSet(),
  * then reuse the learned values if you like with the constructor.
  *
  * This classifier is useful for binary inputs.
  * For continuous (Double) inputs, use GaussianNaiveBayes.
  *
  * Ref: https://en.wikipedia.org/wiki/Naive_Bayes_classifier
  *
  * @param probabilities Map by class: sequence of probabilities for each parameter.
  */
class BernoulliNaiveBayes(val probabilities: Map[Int, Seq[Double]]) {

  /** Predicts the most likely class given the parameters. */
  def predict(parameters: Seq[Int]): Int =
    classLikelihoods(parameters).toSeq.maxBy(_._2)._1

  /**
    * We don't need the exact probabilities here, only some likelihood weights
    * for picking the most likely class. So we take the probability density values,
    * and add them in log space instead of multiplying, to avoid type underflow
    * from multiplying many small fractional values.
    *
    * @return Map(class -> likelihood): weights for belonging to each class given the parameters.
    */
  def classLikelihoods(parameters: Seq[Int]): Map[Int, Double] = {
    require(parameters.size == probabilities.head._2.size, "The number of parameters " +
      s"(${parameters.size}) must match the known set size (${probabilities.size})")

    probabilities.mapValues { parameterStats =>
      (for ((x, p) <- parameters zip parameterStats) yield {
        math.log(Stat.bernoulliDensity(x, p))
      }).sum
    }
  }
}
object BernoulliNaiveBayes {
  /**
    * @param data A sequence of entries: (class: Int, parameters: Seq[Int])
    */
  def fromTrainingSet(data: Seq[(Int, Seq[Int])]): BernoulliNaiveBayes = {

    // We build a transposed matrix of examples for each pixel, grouped by class,
    // to calculate the probabilities.

    val bitwiseByClass = mutable.Map[Int, Seq[mutable.Buffer[Int]]]()

    @tailrec
    def addExamples(data: Seq[(Int, Seq[Int])]): Unit = {
      if (data.nonEmpty) {
        val (label, example) = data.head
        val forClass = bitwiseByClass.getOrElseUpdate(label, Seq.fill(example.length)(mutable.Buffer()))
        forClass zip example map Function.tupled(_ += _)
        addExamples(data.tail)
      }
    }

    addExamples(data)

    def probabilities = bitwiseByClass.mapValues(_.map(Stat.probabilityOfOne)).toMap

    new BernoulliNaiveBayes(probabilities)
  }
}