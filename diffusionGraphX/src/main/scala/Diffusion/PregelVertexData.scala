package Diffusion

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

  // g_t
  //var g_t = g_t_c.dup()

  // A_t
  var At: DoubleMatrix = g_t.dup()

  // phi_tt of edges + half of the edge attribute
  var g_t_phi: DoubleMatrix = DoubleMatrix.zeros(noLabels)
  //for bound computation
  var weight: Double = 0.0

  def +(that: PregelVertexData) = {

    for ((k, v) <- that.g_tt_phi) {
      if (this.g_tt_phi.contains(k) && (v.norm2() < g_tt_phi(k).norm2())) {
        g_tt_phi(k) = v
      }
      this.g_tt_phi += ((k, v))
    }

    for ((k, v) <- that.min_gtt_phi) {
      if (this.min_gtt_phi.contains(k) && (v.norm2() < min_gtt_phi(k).norm2())) {
        min_gtt_phi(k) = v
      }
      this.min_gtt_phi += ((k, v))
    }

    this
  }

  // copy constructor
  def this(copyObject: PregelVertexData) {
    this(copyObject.g_t.dup(), copyObject.g_tt.dup(), copyObject.white, copyObject.gridWidth)
    this.g_tt_phi = copyObject.g_tt_phi.clone()
    this.phi_tt = copyObject.phi_tt.clone()
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