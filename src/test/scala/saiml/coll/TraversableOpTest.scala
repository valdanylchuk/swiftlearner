package saiml.coll

import org.specs2.mutable.Specification
import saiml.coll.TraversableOp._

class TraversableOpTest extends Specification {
  "TraversableOp.repeat" should {
    "repeat a sequence n times" >> {
      Seq(1, 2, 3).repeat(3) must_== Seq(1, 2, 3, 1, 2, 3, 1, 2, 3)
    }
  }
}
