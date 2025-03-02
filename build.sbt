name := "swiftlearner"

organization := "com.danylchuk"

version := "0.2.7"

scalaVersion := "2.11.12"

resolvers ++= Seq(
  "scalaz-bintray" at "https://dl.bintray.com/scalaz/releases"
)

libraryDependencies ++= Seq(
  "com.typesafe.scala-logging" %% "scala-logging"   % "3.4.0",
  "ch.qos.logback"             %  "logback-classic" % "1.1.7",
  "org.specs2"                 %% "specs2-core"     % "3.8.4" % "test"
)

parallelExecution in Test := false

publishMavenStyle := true

publishTo := {
  val nexus = "https://oss.sonatype.org/"
  if (isSnapshot.value)
    Some("snapshots" at nexus + "content/repositories/snapshots")
  else
    Some("releases"  at nexus + "service/local/staging/deploy/maven2")
}

publishArtifact in Test := false

pomIncludeRepository := { _ => false }

pomExtra := (
  <url>https://github.com/valdanylchuk/swiftlearner</url>
    <licenses>
      <license>
        <name>BSD-style</name>
        <url>http://www.opensource.org/licenses/bsd-license.php</url>
        <distribution>repo</distribution>
      </license>
    </licenses>
    <scm>
      <url>git@github.com:valdanylchuk/swiftlearner.git</url>
      <connection>scm:git:git@github.com:valdanylchuk/swiftlearner.git</connection>
    </scm>
    <developers>
      <developer>
        <id>valdanylchuk</id>
        <name>Valentyn Danylchuk</name>
        <url>http://danylchuk.com</url>
      </developer>
    </developers>)