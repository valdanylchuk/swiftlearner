package saiml.math

import org.specs2.mutable.Specification


class StatTest extends Specification {
  "Stat" should {
    "calculate the mean correctly" >> {
      Stat.mean(Seq(0.8, 1.2)) must beCloseTo(1.0 +/- 0.0001)
      Stat.mean(Seq(1.9, 2.1)) must beCloseTo(2.0 +/- 0.0001)
    }

    "calculate the variance correctly" >> {
      Stat.variance(Seq(0.8, 1.2)) must beCloseTo(0.04 +/- 0.0001)
      Stat.variance(Seq(1.9, 2.1)) must beCloseTo(0.01 +/- 0.0001)
    }

    "calculate the standard deviation correctly" >> {
      Stat.stdDev(Seq(0.8, 1.2)) must beCloseTo(0.2 +/- 0.0001)
      Stat.stdDev(Seq(1.9, 2.1)) must beCloseTo(0.1 +/- 0.0001)
    }

    "calculate Gaussian probability density correctly" >> {
      val normalDistributionMax = 1 / math.sqrt(2 * math.Pi)
      Stat.gaussianDensity(0, 0, 1) must beCloseTo(normalDistributionMax +/- 0.0001)
      Stat.gaussianDensity(1, 1, 1) must beCloseTo(normalDistributionMax +/- 0.0001)
      Stat.gaussianDensity(0, 0, 2) must beCloseTo(0.2821 +/- 0.0001)
      Stat.gaussianDensity(0, 1, 2) must beCloseTo(0.2197 +/- 0.0001)
      Stat.gaussianDensity(1, 0, 2) must beCloseTo(0.2197 +/- 0.0001)
      Stat.gaussianDensity(1, 0, 1) must beCloseTo(0.2420 +/- 0.0001)
    }
  }
}