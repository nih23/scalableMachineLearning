package Diffusion

import org.apache.spark.graphx._


/**
  * first draft of scalable diffusion algorithm based on Apache Spark GraphX
  *
  * @author Benjamin Naujoks, Nico Hoffmann
  */

object DiffusionPartitioner extends PartitionStrategy {
	def getPartition(src: VertexId, dst: VertexId, numParts: PartitionID): PartitionID = {
		val res: PartitionID = src.toInt 
		println("called")
		//res % numParts		
		0
	}
}
