package tools

/**
  * Created by nico on 27.07.2016.
  */

import javax.swing.tree.DefaultMutableTreeNode

import ncsa.hdf.`object`._
import ncsa.hdf.`object`.h5.H5ScalarDS
import ncsa.hdf.hdf5lib.{H5, HDF5Constants}
import ncsa.hdf.`object`.{Dataset, Group}
import org.apache.spark._
import org.apache.spark.rdd._
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed._


object HDF5Reader {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("SparkMatrixTest").setMaster("local")
    val sc = new SparkContext(conf)
    val fname: String = "/home/nico/SparkDevel/SSCL/color-seg-n4/clownfish-small.h5"
    var mat = readMatrix(fname, "gm/factors", sc)
    var mat2 = readMatrix(fname, "gm/header", sc)
    var mat3 = readMatrix(fname, "gm/numbers-of-states", sc)
    val noRows = mat.numRows()
    val noCols = mat.numCols()
    //mat = mat.transpose.multiply(mat)
    val matl = mat.toLocalMatrix()
    //val matl2 = mat2.toLocalMatrix()
    val matl3 = mat3.toLocalMatrix()
    println()
  }


  def readMatrix(fname: String, varname: String, sc: SparkContext): BlockMatrix = {
    val rdd = readHDF5IntoRDD(fname, varname, sc)
    println("preparing transformation into blockmatrix")
    val rdd2 = rdd.zipWithIndex().map(f => new IndexedRow(f._2, new DenseVector(f._1)))
    println("transforming into distributed blockmatrix")
    new IndexedRowMatrix(rdd2).toBlockMatrix()
  }


  def getNoRows(fname: String, varName: String): Long = {
    val fname2 = fname.replace("file:", "")
    val fileFormat = FileFormat.getFileFormat(FileFormat.FILE_TYPE_HDF5)
    val testFile = fileFormat.createInstance(fname2, FileFormat.WRITE)
    testFile.open()
    val grp = testFile.getRootNode().asInstanceOf[(javax.swing.tree.DefaultMutableTreeNode)].getUserObject().asInstanceOf[Group]
    val dset = grp.getFileFormat.get("/" + varName).asInstanceOf[H5ScalarDS]
    dset.init()
    val dt: Datatype = dset.getDatatype
    val sizes = dset.getSelectedDims
    testFile.close()
    sizes(1)
  }


  def readHDF5IntoRDD(fname: String, varName: String, sc: SparkContext): RDD[Array[Double]] = {
    val fileFormat = FileFormat.getFileFormat(FileFormat.FILE_TYPE_HDF5)
    if (fileFormat == null) {
      println("[ERR] cant load fileformat!")
      return null
    }

    val testFile = fileFormat.createInstance(fname, FileFormat.WRITE)
    testFile.open()
    val res = testFile.canRead
    val grp = testFile.getRootNode().asInstanceOf[(javax.swing.tree.DefaultMutableTreeNode)].getUserObject().asInstanceOf[Group]
    val dset = grp.getFileFormat.get("/" + varName).asInstanceOf[H5ScalarDS]
    dset.init()
    val dt: Datatype = dset.getDatatype
    val sizes = dset.getSelectedDims
    //sizes(1) = 10000 // DEBUG: REMOVE ME
    // use correct data type
    println("read hdf5 data")
    var data: Array[Double] = null
    var d2: AnyRef = null
    dt.getDatatypeClass match {
      case HDF5Constants.H5T_FLOAT => data = dset.read().asInstanceOf[Array[Float]].map(f => f.toDouble)
      case HDF5Constants.H5T_NATIVE_DOUBLE => data = dset.read().asInstanceOf[Array[Double]]
      case other => d2 = dset.read()
    }

    // fallback in case of long/ulong/[u]int[64] arrays...
    if (d2.isInstanceOf[Array[Long]]) {
      data = d2.asInstanceOf[Array[Long]].map(f => f.toDouble)
    }

    testFile.close()
    println("create RDD")
    val dims = dset.getDims
    if (dims.size == 2) {
      // matrix
      var splitSize: Int = dims(1).toInt
      splitSize = sizes(1).toInt
      sc.parallelize(data.sliding(splitSize, splitSize).toIndexedSeq)
    } else if (dims.size == 1) {
      // vector
      sc.parallelize(Seq(data))
    } else {
      null
    }
  }
}
