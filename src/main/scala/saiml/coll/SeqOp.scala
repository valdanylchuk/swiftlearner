package saiml.coll

import scala.language.implicitConversions
import scala.util.Random


class SeqOp[A](xs: Seq[A]) {
  /** Random sample without repetitions */
  def randomSample(n: Int): Seq[A] = Stream.continually(Random.nextInt(xs.size))
    .distinct.take(n).map(xs).toSeq
}
object SeqOp {
  implicit def seqToSeqOp[A](xs: Seq[A])
  : SeqOp[A] = new SeqOp(xs)
}