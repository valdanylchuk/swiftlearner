package saiml.coll

import org.specs2.mutable.Specification
import saiml.coll.SeqOp._


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