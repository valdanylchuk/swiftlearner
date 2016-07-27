package saiml.coll

import scala.language.implicitConversions


class TraversableOp[A](xs: Traversable[A]) {
  def repeat(times: Int): Traversable[A] =
    Iterator.fill(times)(xs).flatten.toTraversable
}
object TraversableOp {
  implicit def traversableToTraversableOp[A](xs: Traversable[A])
  : TraversableOp[A] = new TraversableOp(xs)
}