package com.danylchuk.swiftlearner.math

/** Statistic helper functions */
object Stat {
  def mean(xs: Seq[Double]): Double = xs.sum / xs.size

  def variance(xs: Seq[Double]): Double = variance(xs, mean(xs))

  /** @param mean The mean value if we already know it; otherwise use variance(xs). */
  def variance(xs: Seq[Double], mean: Double): Double =
    xs.map(_ - mean).map(x => x * x).sum / xs.size

  /** Standard deviation */
  def stdDev(xs: Seq[Double]): Double = math.sqrt(variance(xs))

  /** Gaussian probability density for a parameter with given mean and variance at point x */
  def gaussianDensity(x: Double, mean: Double, variance: Double): Double = {
    val dev = x - mean
    val exponent = math.exp(-(dev * dev / (2 * variance)))
    1 / math.sqrt(2 * math.Pi * variance) * exponent
  }
}