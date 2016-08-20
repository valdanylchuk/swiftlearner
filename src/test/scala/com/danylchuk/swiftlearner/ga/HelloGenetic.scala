package com.danylchuk.swiftlearner.ga

import scala.util.Random


/**
  * Example user domain implementation of Individual.
  *
  * Usage: val result = new Genetic[HelloGenetic](50, 10).optimize(100, 10000L)
  *
  * Ref.: http://www.electricmonk.nl/log/2011/09/28/evolutionary-algorithm-evolving-hello-world/
  */
class HelloGenetic(override val genome: Option[String]) extends Individual[HelloGenetic, String] {
  val hello = HelloGenetic.Hello
  val genomeVal = genome.get

  /** Random default constructor */
  def this() = this(Some(Random.alphanumeric.take(HelloGenetic.Hello.length).mkString))

  /** Fitness function: sum of differences, squared to penalize severe defects */
  override val fitness: Double = {
    val diffs = for ((a, b) <- genomeVal zip hello) yield { val d = a - b; d * d }
    diffs.sum
  }

  /** Crossover: take the head of one genome and the tail of another */
  override def crossover(that: HelloGenetic): HelloGenetic = {
    val pos = Random.nextInt(hello.length)
    val newGenome = this.genomeVal.substring(0, pos) + that.genomeVal.substring(pos)
    new HelloGenetic(Some(newGenome))
  }

  /** Mutate: slightly modify at a random location */
  override def mutate(): HelloGenetic = {
    val pos = Random.nextInt(hello.length)
    val change = Random.nextInt(11) - 5  // [-5 .. +5]
    val newGenome = this.genomeVal.substring(0, pos) +
        (genomeVal.charAt(pos) + change).toChar +
        this.genomeVal.substring(pos+1)
    new HelloGenetic(Some(newGenome))
  }
}
object HelloGenetic {
  val Hello = "Hello, World!"
}