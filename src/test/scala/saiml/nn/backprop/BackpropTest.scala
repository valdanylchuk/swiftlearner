package saiml.nn.backprop

import com.typesafe.scalalogging.LazyLogging
import org.specs2.execute.Result
import org.specs2.mutable.Specification
import saiml.coll.TraversableOp._

class BackpropTest extends Specification with LazyLogging {
  def learnAndTest(nn: BackpropNet, examples: Seq[(Vector[Double], Vector[Double])], times: Int) = {
    val learned = nn.learnSeq(examples.repeat(times))
    Result.foreach(examples) { example =>
      learned.calculateOutput(example._1)
        .head must beCloseTo(example._2.head +/- 0.1)
    }
  }

  "BackpropNet" should {
    "reproduce the known example" >> {
      val input = Vector[Double](0.35, 0.9)
      val target = Vector(0.5)

      val nn = new BackpropNet(
        Vector(Node(Vector(0.1, 0.8)), Node(Vector(0.4, 0.6))),
        Vector(Node(Vector(0.3, 0.9))))

      val learned = nn.learn(input, target)

      learned.hiddenLayer(0).output must beCloseTo(0.6803 +/- 0.0001)
      learned.hiddenLayer(1).output must beCloseTo(0.6637 +/- 0.0001)
      learned.outputLayer(0).output must beCloseTo(0.6903 +/- 0.0001)

      learned.outputLayer(0).error must beCloseTo(-0.0406 +/- 0.0001)
      learned.outputLayer(0).weights(0) must beCloseTo(0.2723 +/- 0.0001)
      learned.outputLayer(0).weights(1) must beCloseTo(0.8730 +/- 0.0001)

      learned.hiddenLayer(0).error must beCloseTo(-0.0025 +/- 0.0002)
      learned.hiddenLayer(0).weights(0) must beCloseTo(0.0991 +/- 0.0002)
      learned.hiddenLayer(0).weights(1) must beCloseTo(0.7977 +/- 0.0002)

      learned.hiddenLayer(1).error must beCloseTo(-0.008 +/- 0.0002)
      learned.hiddenLayer(1).weights(0) must beCloseTo(0.3972 +/- 0.0002)
      learned.hiddenLayer(1).weights(1) must beCloseTo(0.5927 +/- 0.0002)

      learned.calculateOutput(input)(0) must beCloseTo(0.682 +/- 0.0001)
    }

    "learn the OR function" >> {
      val OrExamples = Seq(
        (Vector[Double](0, 0), Vector(0.0)),
        (Vector[Double](0, 1), Vector(1.0)),
        (Vector[Double](1, 0), Vector(1.0)),
        (Vector[Double](1, 1), Vector(1.0)))

      val nn = BackpropNet.randomNet(2, 2, 1, Some(100L))
      learnAndTest(nn, OrExamples, 1000)
    }

    "learn the AND function" >> {
      val AndExamples = Seq(
        (Vector[Double](0, 0), Vector(0.0)),
        (Vector[Double](0, 1), Vector(0.0)),
        (Vector[Double](1, 0), Vector(0.0)),
        (Vector[Double](1, 1), Vector(1.0)))

      val nn = BackpropNet.randomNet(2, 3, 1, Some(100L))
      learnAndTest(nn, AndExamples, 1000)
    }

    "learn the XOR function" >> {
      val XorExamples = Seq(
        (Vector[Double](0, 0), Vector(0.0)),
        (Vector[Double](0, 1), Vector(1.0)),
        (Vector[Double](1, 0), Vector(1.0)),
        (Vector[Double](1, 1), Vector(0.0)))

      val nn = BackpropNet.randomNet(2, 4, 1, Some(100L))
      learnAndTest(nn, XorExamples, 6000)
    }
  }
}