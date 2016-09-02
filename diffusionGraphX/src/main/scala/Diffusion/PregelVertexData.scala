package main.scala.Diffusion

import org.jblas.DoubleMatrix

/**
  * Created by nico on 26.07.2016.
  */
class PregelVertexData(var g_t: DoubleMatrix, var g_tt: DoubleMatrix, var white: Int, val gridWidth: Int) extends java.io.Serializable {
  var noLabels = g_t.getRows()
  // vertex id
  var vid: Int = 0
  var iteration: Int = 0
  var label: Int = 0
  var labelForEnergyCompuytation: Int = 0
  //NOT NEEDED var At = DoubleMatrix.zeros(noLabels)
  var min_gtt_phi = scala.collection.mutable.HashMap.empty[Int, DoubleMatrix]
  // min ( g_tt^phi )
  var out_degree: Int = 0
  //var neighbour_ids = new Array[VertexId](4)
  var g_tt_phi = scala.collection.mutable.HashMap.empty[Int, DoubleMatrix]
  // Contains messages to every neighbor
  var phi_tt = scala.collection.mutable.HashMap.empty[Int, DoubleMatrix] // has phi_tt' and phi_t't
  // A_t
  var At: DoubleMatrix = g_t.dup()
  // phi_tt of edges + half of the edge attribute
  var g_t_phi: DoubleMatrix = DoubleMatrix.zeros(noLabels)

  // copy constructor
  def this(copyObject: PregelVertexData) {
    this(copyObject.g_t, copyObject.g_tt, copyObject.white, copyObject.gridWidth)
    this.g_tt_phi = copyObject.g_tt_phi
    this.phi_tt = copyObject.phi_tt
    this.vid = copyObject.vid
    this.iteration = copyObject.iteration
    this.out_degree = copyObject.out_degree
    this.label = copyObject.label
    this.labelForEnergyCompuytation = copyObject.labelForEnergyCompuytation
  }

  def initPhiTT(neighbourIds: Array[Int]): Unit = {
    for (k <- neighbourIds) {
      phi_tt(k) = DoubleMatrix.zeros(noLabels)
    }
  }
}