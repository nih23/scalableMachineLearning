package Diffusion

import org.jblas.DoubleMatrix

/**
  * Created by nico on 26.07.2016.
  */
class EdgeData(val g_tt_c: DoubleMatrix) extends java.io.Serializable
{
  // g_tt equals g_tt(x,x')
  var g_tt = g_tt_c.dup()
  var min_gtt_phi: DoubleMatrix = DoubleMatrix.zeros(g_tt.rows, g_tt.columns)
  //equals min of g_tt_phi
  var g_tt_phi: DoubleMatrix = DoubleMatrix.zeros(g_tt.rows, g_tt.columns)
  var phi_tt = scala.collection.mutable.HashMap.empty[Int,DoubleMatrix]  // has phi_tt' and phi_t't
}