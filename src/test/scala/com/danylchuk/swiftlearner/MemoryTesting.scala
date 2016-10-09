package com.danylchuk.swiftlearner


trait MemoryTesting {
  def countAllocatedRepeat(nTimes: Int)(block: => Any) = {
    synchronized {
      val before = Runtime.getRuntime.freeMemory
      var i = 0
      while (i < nTimes) {
        block
        i += 1
      }
      before - Runtime.getRuntime.freeMemory
    }
  }
}