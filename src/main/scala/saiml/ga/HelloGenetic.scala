package saiml.ga

import scala.util.Random


/**
  * Example user domain implementation of Individual.
  *
  * Usage: val result = new Genetic[HelloGenetic](100, 10).optimize(1000, 10000L)
  *
  * Ref.: http://www.electricmonk.nl/log/2011/09/28/evolutionary-algorithm-evolving-hello-world/
  */
class HelloGenetic(override val genome: String) extends Individual {
  type Self = HelloGenetic
  val hello = HelloGenetic.Hello

  /** Random default constructor */
  def this() = this(Random.alphanumeric.take(HelloGenetic.Hello.length).mkString)

  /** Fitness function: sum of differences, squared to penalize severe defects */
  override val fitness = {
    val diffs = for ((a, b) <- genome zip hello) yield { val d = a - b; d * d }
    diffs.sum
  }

  /** Crossover: take the head of one genome and the tail of another */
  override def crossover(that: Individual): HelloGenetic = {
    val pos = Random.nextInt(hello.length)
    val newGenome = this.genome.substring(0, pos) + that.genome.substring(pos)
    new HelloGenetic(newGenome)
  }

  /** Mutate: slightly modify at a random location */
  override def mutate(): HelloGenetic = {
    val pos = Random.nextInt(hello.length)
    val change = Random.nextInt(11) - 5  // [-5 .. +5]
    val newGenome = this.genome.substring(0, pos) +
        (genome.charAt(pos) + change).toChar +
        this.genome.substring(pos+1)
    new HelloGenetic(newGenome)
  }
}
object HelloGenetic {
  val Hello = "Hello, World!"
}