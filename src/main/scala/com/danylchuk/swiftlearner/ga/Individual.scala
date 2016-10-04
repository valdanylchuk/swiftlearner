package com.danylchuk.swiftlearner.ga

import scala.util.Random

/**
  * You should pass a class derived from this one as a type parameter to Genetic.
  *
  * See the HelloGenetic example in the tests.
  */
abstract class Individual[A <: Individual[A]](val genome: Vector[Double]) {
  /** Please override with a random genome constructor satisfying your domain rules. */
  def this() = this(Vector.empty)

  /** Domain-specific fitness function. 0 = perfect fit; the higher the worse. */
  def fitness: Double

  /**
    *  Domain-specific crossover function. Some ideas:
    *  - copy a fragment
    *  - copy random parts with some probability
    *  - copy one configuration bit while preserving some constraints
    */
  def crossover(that: A): A

  /** A useful typical implementation: take the head of one genome and the tail of another */
  protected def crossoverAtRandomPoint(that: Individual[A]): Vector[Double] = {
    val pos = Random.nextInt(genome.length)
    this.genome.take(pos) ++ that.genome.drop(pos)
  }

  /**
    * Domain-specific mutate function. Some ideas:
    * - change a random element by a random value
    * - switch two random elements
    */
  def mutate(): A
}