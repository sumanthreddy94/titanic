ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.13.12"

lazy val root = (project in file("."))
  .settings(
    name := "titanic-survival",
    idePackagePrefix := Some("edu.neu.coe.csye7200"),
    // https://mvnrepository.com/artifact/org.apache.spark/spark-core
    libraryDependencies += "org.apache.spark" %% "spark-core" % "3.5.0",
    // https://mvnrepository.com/artifact/org.apache.spark/spark-sql
    libraryDependencies += "org.apache.spark" %% "spark-sql" % "3.5.0" % "provided",
    // https://mvnrepository.com/artifact/org.apache.spark/spark-mllib
    libraryDependencies += "org.apache.spark" %% "spark-mllib" % "3.5.0" % "provided"
  )
