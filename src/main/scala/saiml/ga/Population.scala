package saiml.ga

import scala.reflect.ClassTag
import scala.util.Random


/** Selects parents and evolves the next generation */
class Population[A <: Individual[A, B] :ClassTag, B](
  val size: Int,
  tournamentSize: Int,
  givenIndividuals: Option[Vector[A]] = None
) {
  val individuals: Vector[A] = givenIndividuals getOrElse
    Vector.tabulate(size)(_ => implicitly[ClassTag[A]].runtimeClass.newInstance.asInstanceOf[A])

  def getFittest: A = individuals.minBy(_.fitness)

  /** Tournament selection: pick the fittest one out of a random sample */
  def tournamentSelect(): A = Stream.continually(Random.nextInt(individuals.size))
    .distinct.take(tournamentSize).map(individuals).minBy(_.fitness)

  def evolve: Population[A, B] = {
    val nextGen = (1 until size).map { _ =>  // skip one for the elite
      val parent1 = tournamentSelect()
      val parent2 = tournamentSelect()
      parent1.crossover(parent2).mutate()
    }
    val withElite = (getFittest +: nextGen).toVector
    new Population(size, tournamentSize, Some(withElite))
  }
}