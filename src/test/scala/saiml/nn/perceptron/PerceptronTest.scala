package saiml.nn.perceptron

import com.typesafe.scalalogging.LazyLogging
import org.specs2.execute.Result
import org.specs2.mutable.Specification
import saiml.coll.TraversableOp._


class PerceptronTest extends Specification with LazyLogging {
  def learnAndTest(examples: Seq[(Vector[Double], Boolean)], times: Int) = {
    val p = new Perceptron(Vector(0.0, 0.0))
    val learned = p.learn(examples.repeat(times))
    Result.foreach(examples) { example =>
      learned.f(example._1) must_== example._2
    }
  }

  "Perceptron" should {
    "calculate the activation function correctly" >> {
      val p = new Perceptron(Vector(2.0, 3.0), -25.0)
      p.f(Vector(4.0, 5.0)) must beFalse
      p.f(Vector(4.0, 6.0)) must beTrue
    }

    "learn the AND function" >> {
      val AndExamples = Seq(
        (Vector[Double](0, 0), false),
        (Vector[Double](0, 1), false),
        (Vector[Double](1, 0), false),
        (Vector[Double](1, 1), true))

      learnAndTest(AndExamples, 10)
    }

    "learn the OR function" >> {
      val OrExamples = Seq(
        (Vector[Double](0, 0), false),
        (Vector[Double](0, 1), true),
        (Vector[Double](1, 0), true),
        (Vector[Double](1, 1), true))

      learnAndTest(OrExamples, 10)
    }
  }
}