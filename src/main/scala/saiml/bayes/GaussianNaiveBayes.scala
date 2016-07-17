package saiml.bayes

import saiml.math.Stat


/**
  * Gaussian naive Bayes classifier.
  *
  * You can create it first time with fromTrainingSet(),
  * then reuse the learned values if you like with the constructor.
  *
  * Ref: https://en.wikipedia.org/wiki/Naive_Bayes_classifier
  *      http://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
  *
  * @param meansAndVariances Map by class: sequence of (mean, variance) for each parameter.
  */
class GaussianNaiveBayes(val meansAndVariances: Map[Int, Seq[(Double, Double)]]) {

  /** Predicts the most likely class given the parameters. */
  def predict(parameters: Seq[Double]): Int =
    classLikelihoods(parameters).toSeq.maxBy(_._2)._1

  /**
    * We don't need the exact probabilities here, only some likelihood weights
    * for picking the most likely class. So we take the probability density values,
    * and add them in log space instead of multiplying, to avoid type underflow
    * from multiplying many small fractional values.
    *
    * @return Map(class -> likelihood): weights for belonging to each class given the parameters.
    */
  def classLikelihoods(parameters: Seq[Double]): Map[Int, Double] = {
    require(parameters.size == meansAndVariances.head._2.size, "The number of parameters " +
      s"(${parameters.size}) must match the known set size (${meansAndVariances.size})")

    meansAndVariances.mapValues { parameterStats =>
      (for ((x, stats) <- parameters zip parameterStats) yield {
        math.log(Stat.gaussianDensity(x, stats._1, stats._2))
      }).sum
    }
  }
}
object GaussianNaiveBayes {
  /**
    * @param data A sequence of entries: (class: Int, parameters: Seq[Double])
    */
  def fromTrainingSet(data: Seq[(Int, Seq[Double])]): GaussianNaiveBayes = {
    val byClass: Map[Int, Seq[Seq[Double]]] = data.groupBy(_._1).mapValues(_.map(_._2))

    def meansAndVariances = byClass.mapValues(_.transpose.map { xs =>
      val mean = Stat.mean(xs)
      (mean, Stat.variance(xs, mean))
    })

    new GaussianNaiveBayes(meansAndVariances)
  }
}