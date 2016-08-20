package com.danylchuk.swiftlearner

package object util {
  @inline def byteToUnsignedInt(x: Byte): Int = x.toInt & 0xff
}