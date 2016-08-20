package com.danylchuk.swiftlearner.coll

import com.danylchuk.swiftlearner.coll.SeqOp._
import org.specs2.mutable.Specification


class SeqOpTest extends Specification {
  val testSeq = Seq.tabulate(1000)(identity)

  "SeqOp.randomSample" should {
    "return n items" >> {
      testSeq.randomSample(10) must haveLength(10)
    }
    "return random items" >> {
      testSeq.randomSample(5) must_!= testSeq.randomSample(5)
    }
    "return unique items" >> {
      testSeq.randomSample(1000).distinct must haveLength(1000)
    }
  }
}