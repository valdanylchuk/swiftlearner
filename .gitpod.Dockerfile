FROM gitpod/workspace-full

RUN brew install scala
RUN brew install sbt
RUN brew install scalaenv
RUN scalaenv install scala-2.11.8 && scalaenv global scala-2.11.8
