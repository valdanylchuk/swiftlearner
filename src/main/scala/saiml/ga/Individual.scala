package saiml.ga


/** This should be inherited by the library user */
abstract class Individual(val genome: String) {
  /** Define to match your inherited class */
  type Self  // type system voodoo ref: http://stackoverflow.com/a/14905650

  /** Please override with a random genome constructor satisfying your domain rules. */
  def this() = this("")

  /** Domain-specific fitness function. 0 = perfect fit; the higher the worse. */
  def fitness: Int

  /**
    *  Domain-specific crossover function. Some ideas:
    *  - copy a fragment
    *  - copy random parts with some probability
    *  - copy one configuration bit while preserving some constraints
    */
  def crossover(that: Individual): Self

  /**
    * Domain-specific mutate function. Some ideas:
    * - change a random element by a random value
    * - switch two random elements
    */
  def mutate(): Self
}