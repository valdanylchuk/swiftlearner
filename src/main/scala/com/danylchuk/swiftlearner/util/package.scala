package com.danylchuk.swiftlearner

package object util {
  @inline def byteToUnsignedInt(x: Byte): Int = x.toInt & 0xff

  @inline def byteToBinary(x: Byte, threshold: Int): Int =
    if (byteToUnsignedInt(x) < threshold) 0 else 1
}