package com.danylchuk.swiftlearner.ga

import scala.util.Random


/**
  * Example user domain implementation of Individual.
  *
  * Usage: val result = new Genetic[HelloGenetic](50, 10).optimize(100, 10000L)
  *
  * Ref.: http://www.electricmonk.nl/log/2011/09/28/evolutionary-algorithm-evolving-hello-world/
  */
class HelloGenetic(override val genome: Vector[Double]) extends Individual[HelloGenetic] {
  val hello = HelloGenetic.HelloDouble

  /** Random default constructor */
  def this() = this(Random.alphanumeric.take(HelloGenetic.Hello.length).map(_.toDouble).toVector)

  /** Fitness function: sum of differences, squared to penalize severe defects */
  override val fitness: Double = {
    val diffs = for ((a, b) <- genome zip hello) yield { val d = a - b; d * d }
    diffs.sum
  }

  /** Crossover: take the head of one genome and the tail of another */
  override def crossover(that: HelloGenetic): HelloGenetic =
    new HelloGenetic(crossoverAtRandomPoint(that))

  /** Mutate: slightly modify at a random location */
  override def mutate(): HelloGenetic = {
    val pos = Random.nextInt(hello.length)
    val change = Random.nextInt(11) - 5  // [-5 .. +5]
    val newGenome = this.genome.updated(pos, genome(pos) + change)
    new HelloGenetic(newGenome)
  }
}
object HelloGenetic {
  val Hello = "Hello, World!"
  val HelloDouble = Hello.map(_.toDouble)
}