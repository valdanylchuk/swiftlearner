package com.danylchuk.swiftlearner.nn.backprop

import com.danylchuk.swiftlearner.MemoryTesting
import com.danylchuk.swiftlearner.coll.TraversableOp._
import com.typesafe.scalalogging.LazyLogging
import org.specs2.execute.Result
import org.specs2.mutable.Specification

class BackpropTest extends Specification with LazyLogging with MemoryTesting {
  def learnAndTest(nn: BackpropNet, examples: Seq[(Array[Float], Array[Float])], times: Int) = {
    val learned = nn.learnSeq(examples.repeat(times))
    Result.foreach(examples) { example =>
      learned.calculateOutput(example._1)
        .head must beCloseTo(example._2.head +/- 0.1f)
    }
  }

  val netForMemTest = BackpropNet.randomNet(100, 100, 100)
  val inputForMemTest = Array.tabulate(100)(_.toFloat)

  "BackpropNet" should {
    "reproduce the known example" >> {
      val input = Array[Float](0.35f, 0.9f)
      val target = Array(0.5f)

      val nn = new BackpropNet(
        Array(Node(Array(0.1f, 0.8f)), Node(Array(0.4f, 0.6f))),
        Array(Node(Array(0.3f, 0.9f))))

      val learned = nn.learn(input, target)

      learned.hiddenLayer(0).output must beCloseTo(0.6803f +/- 0.0001f)
      learned.hiddenLayer(1).output must beCloseTo(0.6637f +/- 0.0001f)
      learned.outputLayer(0).output must beCloseTo(0.6903f +/- 0.0001f)

      learned.outputLayer(0).error must beCloseTo(-0.0406f +/- 0.0001f)
      learned.outputLayer(0).weights(0) must beCloseTo(0.2723f +/- 0.0001f)
      learned.outputLayer(0).weights(1) must beCloseTo(0.8730f +/- 0.0001f)

      learned.hiddenLayer(0).error must beCloseTo(-0.0025f +/- 0.0002f)
      learned.hiddenLayer(0).weights(0) must beCloseTo(0.0991f +/- 0.0002f)
      learned.hiddenLayer(0).weights(1) must beCloseTo(0.7977f +/- 0.0002f)

      learned.hiddenLayer(1).error must beCloseTo(-0.008f +/- 0.0002f)
      learned.hiddenLayer(1).weights(0) must beCloseTo(0.3972f +/- 0.0002f)
      learned.hiddenLayer(1).weights(1) must beCloseTo(0.5927f +/- 0.0002f)

      learned.calculateOutput(input)(0) must beCloseTo(0.682f +/- 0.0001f)
    }

    "learn the OR function" >> {
      val OrExamples = Seq(
        (Array[Float](0f, 0f), Array(0.0f)),
        (Array[Float](0f, 1f), Array(1.0f)),
        (Array[Float](1f, 0f), Array(1.0f)),
        (Array[Float](1f, 1f), Array(1.0f)))

      val nn = BackpropNet.randomNet(2, 2, 1, Some(100L))
      learnAndTest(nn, OrExamples, 1000)
    }

    "learn the AND function" >> {
      val AndExamples = Seq(
        (Array[Float](0f, 0f), Array(0.0f)),
        (Array[Float](0f, 1f), Array(0.0f)),
        (Array[Float](1f, 0f), Array(0.0f)),
        (Array[Float](1f, 1f), Array(1.0f)))

      val nn = BackpropNet.randomNet(2, 3, 1, Some(100L))
      learnAndTest(nn, AndExamples, 1000)
    }

    "learn the XOR function" >> {
      val XorExamples = Seq(
        (Array[Float](0f, 0f), Array(0.0f)),
        (Array[Float](0f, 1f), Array(1.0f)),
        (Array[Float](1f, 0f), Array(1.0f)),
        (Array[Float](1f, 1f), Array(0.0f)))

      val nn = BackpropNet.randomNet(2, 4, 1, Some(100L))
      learnAndTest(nn, XorExamples, 6000)
    }

    "use memory sparingly in calculateOutputFor" >> skipped {
      val node = netForMemTest.hiddenLayer(1)
      countAllocatedRepeat(100) {
        node.calculateOutputFor(inputForMemTest)
      } must_== 0L  // sometimes works; OK
    }

    "use memory sparingly in calculateOutput" >> skipped {
      countAllocatedRepeat(1000) {
        netForMemTest.calculateOutput(inputForMemTest)
      } must be_<(20 * 1000000L)
      // 6-12M usual value; room for improvement
      // 100 bytes per iteration
    }

    "use memory sparingly in learn" >> {
      countAllocatedRepeat(100) {
        netForMemTest.learn(inputForMemTest, inputForMemTest)
      } must_== 0L //(10 * 1000000L)
      // 30 - 1M
      // 100 - 2M
      // 200 - 3.5M
      // 300 - 4M
      // 500 - 5M
      // 700 - 5.5M
      // 1000 - 7M
      // 2000 - GC starts
      // ~1.5M + 5.5k per iteration
      // After extracting hiddenLayerOutput: 1000 - 6M
      // nHidden=100 times/iter: hiddenLayer(i).calculateOutputFor(example): ArrayOp.dot
      // nOutput=100 times/iter: outputLayer(j).calculateOutputFor(hiddenLayerOutput): ArrayOp.dot
      // nOutput=100 times/iter: outputLayer(j).updated(hiddenLayerOutput, partialError, rate)
      // nHidden=100 times/iter: hiddenLayer(i).updated(example, partialError, rate)
      // TODO: test calculateOutputFor and updated separately
    }
  }
}