name := "swiftlearner"

organization := "com.danylchuk"

version := "0.2.0"

scalaVersion := "2.11.8"

resolvers ++= Seq(
  "scalaz-bintray" at "http://dl.bintray.com/scalaz/releases"
)

libraryDependencies ++= Seq(
  "com.typesafe.scala-logging" %% "scala-logging"   % "3.4.0",
  "ch.qos.logback"             %  "logback-classic" % "1.1.7",
  "org.specs2"                 %% "specs2-core"     % "3.8.4" % "test"
)

parallelExecution in Test := false