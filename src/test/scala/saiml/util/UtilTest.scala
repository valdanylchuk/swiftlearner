package saiml.util

import org.specs2.mutable.Specification


class UtilTest extends Specification {
  "byteToUnsignedInt" should {
    "get positive value for a number greater than 127" >> {
      val b: Byte = 200.toByte
      b.toInt must_== -56
      byteToUnsignedInt(b) must_== 200
    }
  }
}