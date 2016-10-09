package com.danylchuk.swiftlearner


trait MemoryTesting {
  def countAllocatedRepeat(nTimes: Int)(block: => Any) = {
    synchronized {
      val before = Runtime.getRuntime.freeMemory

      Iterator.range(0, nTimes).foreach { _ =>
        block
      }

      val after = Runtime.getRuntime.freeMemory
      before - after
    }
  }
}