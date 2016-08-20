package com.danylchuk.swiftlearner.coll

import com.danylchuk.swiftlearner.coll.TraversableOp._
import org.specs2.mutable.Specification

class TraversableOpTest extends Specification {
  "TraversableOp.repeat" should {
    "repeat a sequence n times" >> {
      Seq(1, 2, 3).repeat(3) must_== Seq(1, 2, 3, 1, 2, 3, 1, 2, 3)
    }
  }
}
