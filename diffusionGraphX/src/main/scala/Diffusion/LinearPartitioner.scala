package main.scala.Diffusion

import org.apache.spark.graphx._


/**
  * Created by nico on 02.09.16.
  */
object LinearPartitioner extends PartitionStrategy {
  override def getPartition(src: VertexId, dst: VertexId, numParts: PartitionID): PartitionID = {
    val partId = Math.floor(src.toInt / (DiffusionPregel.noNodes / numParts)).toInt
    partId
  }
}
