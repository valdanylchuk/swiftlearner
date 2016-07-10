package saiml.ga

import scala.reflect.ClassTag
import scala.util.Random


/** Selects parents and evolves the next generation */
class Population[A <: Individual {type Self = A} :ClassTag](
  val size: Int,
  tournamentSize: Int,
  givenIndividuals: Option[Vector[A]] = None
) {
  val individuals: Vector[A] = givenIndividuals getOrElse
    Vector.tabulate(size)(_ => implicitly[ClassTag[A]].runtimeClass.newInstance.asInstanceOf[A])

  def getFittest: A = individuals.minBy(_.fitness)

  /** Tournament selection: pick the fittest one out of a random sample */
  def tournamentSelect(): A = {
    Vector.tabulate(tournamentSize) { _ =>
      individuals(Random.nextInt(individuals.size))
    }.minBy(_.fitness)
  }

  def evolve: Population[A] = {
    val nextGen = (1 until size).map { _ =>  // skip one for the elite
      val parent1: A = tournamentSelect()
      val parent2: A = tournamentSelect()
      val child: A = parent1.crossover(parent2)
      child.mutate()
    }
    val withElite = (getFittest +: nextGen).toVector
    new Population(size, tournamentSize, Some(withElite))
  }
}