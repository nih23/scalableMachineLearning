#include "../header/parallelDiffusion.hxx"

#include "../header/reparametrization_mpi.hxx"
//#include "../header/diffusionMPI.hxx"

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/graphicalmodel/space/simplediscretespace.hxx>

//#define EMPLOYEDMPIDT MPI_BYTE
#define TAGOFFSET 10000
//#define EMPLOYEDMPIDT MPI_DOUBLE
//#define USEMPI
//#define DEBUGSEND
#ifdef USEMPI
#include <mpi.h>
#endif



// 32 vs 64bit check
// Check windows
#if _WIN32 || _WIN64
   #if _WIN64
     #define ENV64BIT
  #else
    #define ENV32BIT
  #endif
#endif

// Check GCC
#if __GNUC__
  #if __x86_64__ || __ppc64__
    #define ENV64BIT
  #else
    #define ENV32BIT
  #endif
#endif




template<class GM, class ACC>
std::string ParallelDiffusion<GM, ACC>::name() const
{
	return _name;
}

/*
 *
 * initialize data structures
 *
 */
template<class GM, class ACC>
ParallelDiffusion<GM,ACC>::ParallelDiffusion(GM& gm, size_t maxIterations, ValueType convBound, int nx, int ny) : _gm(gm) {
	_name = "Diffusion MPI";
	_griddx = nx;
	_griddy = ny;

#ifdef USEMPI
	MPI_Comm_rank(MPI_COMM_WORLD, &_mpi_myRank);
	MPI_Comm_size(MPI_COMM_WORLD, &_mpi_commSize);
	std::cout << "Hello from MPI @ rank " << _mpi_myRank << " / " << _mpi_commSize << std::endl;
#else
	std::cout << "MPI disabled." << std::endl;
	_mpi_myRank = 0;
	_mpi_commSize = 1;
#endif

	// project primal factor space into variable space for easy distribution by mpi
	for (IndexType factorIndex=0; factorIndex<_gm.numberOfFactors(); ++factorIndex)
	{
		const typename GM::FactorType& factor = _gm[factorIndex];
		IndexType noLabels = factor.numberOfLabels(0); // left variable
		IndexType noVariables = factor.numberOfVariables();
		IndexType leftVar = factor.variableIndex(0);

		// new: initialize dual vars for non-unary factors
		if(noVariables > 1)
		{
			UnaryFactor dualVarsForFactor;
			dualVarsForFactor.resize(noLabels);
			_phiTT[factorIndex] = dualVarsForFactor;
		}

		VecFactorIndices got;
		VecNeighbourIndices gotNgs;
		MapFactorindexDualvariable gotFIDual;

		got = _variableToFactors[leftVar];
		got.push_back(factorIndex);

		gotFIDual = _dualVars[leftVar];       


		UnaryFactor uf;
		uf.resize(noLabels);
		gotFIDual[factorIndex] = uf;

		_dualVars[leftVar] = gotFIDual;

		if(noVariables > 1) {
			IndexType rightVar = factor.variableIndex(1);
			UnaryFactor uf2;
			uf2.resize(_gm.numberOfLabels(rightVar));
			gotFIDual = _dualVars.at(rightVar);
			gotFIDual[factorIndex] = uf2;
			_dualVars[rightVar] = gotFIDual;
			// insert var + factor into dual variables storage
			gotNgs = _variableToNeighbours[leftVar];
			gotNgs.push_back(rightVar);
			_variableToNeighbours[leftVar] = gotNgs;


			VecFactorIndices got2 = _variableToFactors[rightVar];
			got2.push_back(factorIndex);
			_variableToFactors[rightVar] = got2;
		}

		_variableToFactors[leftVar] = got;
	}

	_gPhis = std::vector<ValueType>(_gm.numberOfVariables());
	_states = std::vector<LabelType>(_gm.numberOfVariables());
	_weights = std::vector<ValueType>(_gm.numberOfVariables());
	_energy = static_cast<ValueType>(0);
	_bound = static_cast<ValueType>(0);

	this->_maxIterations = maxIterations;
	this->_convBound = convBound;
	//maxIterations_(maxIterations), convBound_(convBound), variant_(variant), weights_(std::vector<ValueType>(gm_.numberOfVariables())), gridSizeX_(gridSizeX), gridSizeY_(gridSizeY), bound_(0.), name_("Diffusion")


	IndexType numberFac = _gm.numberOfFactors(0);
	for (auto i=1; i<_gm.numberOfVariables(); ++i)
	{
		if (_gm.numberOfFactors(i) == numberFac)
		{
			_lastColumnId = i;
			break;
		}
	}
	std::cout << "LCid: " << _lastColumnId << std::endl;

}

template<class GM, class ACC>
bool ParallelDiffusion<GM,ACC>::blackAndWhite(IndexType index)
{
	return (((index%_lastColumnId)+(index/_lastColumnId))%2==0);	  
}

template<class GM, class ACC>
const GM& ParallelDiffusion<GM,ACC>::graphicalModel() const
{
	return _gm;
}

template <class GM, class ACC>
void ParallelDiffusion<GM,ACC>::computeWeights()
{
	for (auto varIndex=0; varIndex<_gm.numberOfVariables(); ++varIndex)
	{
		_weights[varIndex] = 1./((ValueType)_gm.numberOfFactors(varIndex)); // w_0 = 0
	}
}

/*
 *
 * infer new model state
 *
 */
template<class GM, class ACC>
opengm::InferenceTermination ParallelDiffusion<GM,ACC>::infer()
{ 
	// initialize weight vectors
	this->computeWeights();

	for(size_t i=0;i<this->_maxIterations;++i)
	{
		auto before = std::time(nullptr);
		//std::cout << "Diffusion Iteration " << i << std::endl;

		// black
		//std::cout << "1. update dual variables (black fields) R" << _mpi_myRank << std::endl;//}
		this->diffusionIterationNew(true);

//		this->receiveMPIUpdatesOfDualVariables(true);

		//std::cout << "2. update dual variables (white fields) R" << _mpi_myRank << std::endl;//}
		this->diffusionIterationNew(false);

//		this->receiveMPIUpdatesOfDualVariables(false);

		this->updateAllStates();
		auto now = std::time(nullptr);

		std::cout << "dt (phi) " << (now-before) << std::endl;

		_bound = this->computeBound(); // optimize this function as well! since it slows down computational time remarkedly

		now = std::time(nullptr);

		_energy = this->computeEnergy();
		auto oldBound = _bound;

	}
	return opengm::NORMAL;
}

template<class GM, class ACC>
typename GM::ValueType ParallelDiffusion<GM,ACC>::computeEnergy() {
	int sz;
	std::pair<IndexType, IndexType> bds;

	// this->updateAllStates();

	// send local state update to master for energy computation
	// MPI_UINT64_T
	if(_mpi_myRank > 0) {
		bds = this->computePartitionBounds();
#ifdef ENV64BIT
		uint64_t* begin = &_states[bds.first];
#else
		uint32_t* begin = &_states[bds.first];
#endif
		sz = bds.second - bds.first + 1;
#ifdef USEMPI
		MPI_Request myreq;
#endif
#ifdef DEBUGSEND
		std::cout << "rank " << _mpi_myRank << " sending " << sz << " states to master " << " sample: " << _states[bds.first] << std::endl;
#endif
		//      std::cout << _states[bds.first + 1] << " " << _states[bds.first + 2] << " " << _states[bds.first + 3] << " -> ";
#ifdef USEMPI
	#ifdef ENV64BIT
		MPI_Send(begin, sz, MPI_UINT64_T, 0, 42+_mpi_myRank, MPI_COMM_WORLD);
	#else
		MPI_Send(begin, sz, MPI_UINT32_T, 0, 42+_mpi_myRank, MPI_COMM_WORLD);
	#endif
#endif
		return 0;
	}

#ifdef DEBUGSEND
	std::cout << "[d] receiving state updates from children." << std::endl;
#endif

#ifdef USEMPI
	for(int i = 1; i < _mpi_commSize; i++) {
		MPI_Status status;
		bds = this->computePartitionBounds(i);
		sz = bds.second - bds.first + 1;
		std::vector<uint64_t> rcvBuffer;
		rcvBuffer.resize(sz);

		//   MPI_Recv(&_states[bds.first], sz, MPI_UINT64_T, i, 42, MPI_COMM_WORLD, NULL);
#ifdef ENV64BIT
		MPI_Recv(&rcvBuffer[0], sz, MPI_UINT64_T, i, 42+i, MPI_COMM_WORLD, &status);
#else
		MPI_Recv(&rcvBuffer[0], sz, MPI_UINT32_T, i, 42+i, MPI_COMM_WORLD, &status);
#endif

		for(int k = 0; k < sz; k++)
		{
			_states[bds.first+k] = rcvBuffer[k];
		}

#ifdef DEBUGSEND
		std::cout << "[d] received data from child " << i << " sample: " << _states[bds.first] << ";" << _states[bds.second] << std::endl;
#endif
		//    std::cout << _states[bds.first + 1] << " " << _states[bds.first + 2] << " " << _states[bds.first + 3] << std::endl;
	}
#endif

	return _gm.evaluate(_states.begin());
}

template<class GM, class ACC>
int ParallelDiffusion<GM,ACC>::convertVariableIndexToMPIRank(IndexType variableIndex) {
	int partitionSz = _gm.numberOfVariables() / _mpi_commSize;
	return variableIndex / partitionSz;
}

template<class GM, class ACC>
void ParallelDiffusion<GM,ACC>::outputDualvars() {

	int noVars = _gm.numberOfVariables();
	for(int i = 0; i < noVars; i++) {
		VecFactorIndices factorsOfVar = _variableToFactors[i];
		for(IndexType fi : factorsOfVar) {
			std::cout << i << ";" << fi << " ; ";
			UnaryFactor uf = _dualVars[i][fi];
			for(ValueType uf_value : uf) {
				std::cout << uf_value << " ; ";
			}
			std::cout << std::endl;
		}

	}
}

template<class GM, class ACC>
void ParallelDiffusion<GM,ACC>::receiveMPIUpdatesOfDualVariables(bool isBlack)
{
	auto checkerboardOffset = isBlack ? 0 : 1;
	std::pair<IndexType, IndexType> partitionBds = this->computePartitionBounds(); 
	//  for (IndexType myVariableIndex=partitionBds.first + checkerboardOffset; myVariableIndex<partitionBds.second; myVariableIndex+=2) //or i<gm_.numberOfVariables()
	for (IndexType myVariableIndex=partitionBds.first; myVariableIndex<partitionBds.second; myVariableIndex++) //or i<gm_.numberOfVariables()
	{
		if(this->blackAndWhite(myVariableIndex) != isBlack) {
			continue;
		}
		VecFactorIndices factorsOfVariable;
		factorsOfVariable = _variableToFactors[myVariableIndex];
		for(IndexType factIdx : factorsOfVariable)
		{
			FactorType factor = _gm[factIdx];

			/*
	if(_gm[factIdx].variableIndex(0) == myVariableIndex)
	{
	  continue;
	}*/

			// only care about pairwise potentials
			if (factor.numberOfVariables() == 1)
			{
				continue;
			}

			// create neighbour data structures for current factor
			std::vector<IndexType> neighboursOfDualUpdates;
			for(int j = 0; j < factor.numberOfVariables(); j++) {
				if(factor.variableIndex(j) == myVariableIndex) {
					continue;
				}  
				neighboursOfDualUpdates.push_back(factor.variableIndex(j));
			}

			// we only rcv data starting from the 1.5th iteration
			// receive data from neighbours
			for(IndexType neighbourIdx : neighboursOfDualUpdates) {

				// minimize communication overhead
				if(convertVariableIndexToMPIRank(neighbourIdx) == _mpi_myRank) {
					continue;
				}



				std::vector<double> rcvBuffer; // = _dualVars[neighbourIdx][factIdx];
				rcvBuffer.resize(_dualVars[neighbourIdx][factIdx].size());

				//std::cout << "receiving " << factIdx << " from " << convertVariableIndexToMPIRank(neighbourIdx) << std::endl;
#ifdef USEMPI
				MPI_Status status;	  
				MPI_Recv(&(rcvBuffer[0]), rcvBuffer.size(), EMPLOYEDMPIDT, convertVariableIndexToMPIRank(neighbourIdx), factIdx+myVariableIndex*TAGOFFSET, MPI_COMM_WORLD, &status);
#endif

				std::stringstream ss;
				ss << "[";
				for(int l = 0; l < rcvBuffer.size(); ++l) {
					ss << rcvBuffer[l] << " ";
					_dualVars[neighbourIdx][factIdx][l] = rcvBuffer[l];
				}
				ss << "]";
				std::cout << " variable " << myVariableIndex << "(black: " << this->blackAndWhite(myVariableIndex) << ", rank " << _mpi_myRank << ") received " << factIdx << " from " << neighbourIdx << " (rank" << convertVariableIndexToMPIRank(neighbourIdx) << "). Data: " << ss.str() << std::endl;


			}
		}
	}
}

template<class GM, class ACC>
typename ParallelDiffusion<GM, ACC>::UnaryFactor ParallelDiffusion<GM,ACC>::computeAt(IndexType variableIndex) {

		UnaryFactor At;
		IndexType noLabels = _gm.numberOfLabels(variableIndex);

		At.resize(noLabels);

		for(IndexType lblId = 0; lblId < noLabels; lblId++)
		{
			At.push_back( SumNeighboursMinGttPhi(variableIndex, lblId) + getValueOfUnaryFactor(variableIndex, lblId) );
		}

		return At;
	}

template<class GM, class ACC>
	double ParallelDiffusion<GM,ACC>::SumNeighboursMinGttPhi(IndexType variableIndex, LabelType xt) {

		IndexType noFactors = _gm.numberOfFactors(variableIndex);
		double res = 0;

		// iterate over all neighbours of variableIndex and sum minimum edges
		for (IndexType neighbFactIdx = 0; neighbFactIdx < noFactors; neighbFactIdx++)
		{
			IndexType factorIdx = _gm.factorOfVariable(variableIndex, neighbFactIdx);
			FactorType factor = _gm[factorIdx];

			// we only care about pairwise terms
			if (_gm.numberOfVariables(factorIdx) == 1)
			{
				continue;
			}

			IndexType leftVarId = factor.variableIndex(0);
			IndexType rightVarId = factor.variableIndex(1);
			IndexType adjacentVarId;

			if(leftVarId != variableIndex)
			{
				adjacentVarId = leftVarId;
			} else {
				adjacentVarId = rightVarId;
			}

			// find minimum edge gtt^phi over all labels at adjacent node

			res += computeMinGttPhi(variableIndex, factorIdx, xt);
		}

		return res;
	}

template<class GM, class ACC>
typename ParallelDiffusion<GM, ACC>::ValueType ParallelDiffusion<GM,ACC>::computeMinGttPhi(IndexType varIdx, IndexType factorIndex, LabelType xt) {

		bool switch_xt_xtt = _gm[factorIndex].variableIndex(1) == varIdx;
		IndexType adjacentVarId;
		if(switch_xt_xtt == false) {
			adjacentVarId = _gm[factorIndex].variableIndex(1);
		} else {
			adjacentVarId = _gm[factorIndex].variableIndex(0);
		}
		ValueType gttPhiMin = computeGttPhi(factorIndex,xt,0,switch_xt_xtt);

		for (LabelType lblId = 1; lblId < _gm.numberOfLabels(adjacentVarId); lblId++ )
		{
			ValueType gttPhi = computeGttPhi(factorIndex,xt,lblId,switch_xt_xtt);
			if(gttPhi < gttPhiMin) {
				gttPhiMin = gttPhi;
			}
		}

		return gttPhiMin;
	}

	/*
 	 *
 	 */

template<class GM, class ACC>
typename ParallelDiffusion<GM, ACC>::ValueType ParallelDiffusion<GM,ACC>::computeGttPhi(IndexType factorIndex, LabelType xt, LabelType xtt, bool switch_xt_xtt) {

		FactorType factor = _gm[factorIndex];

		IndexType leftVarId = _gm[factorIndex].variableIndex(0);
		ValueType leftMessage = _dualVars[leftVarId][factorIndex][xt];

		IndexType rightVarId = _gm[factorIndex].variableIndex(1);
		ValueType rightMessage = _dualVars[rightVarId][factorIndex][xtt];

		LabelType factorLbls[2] = {xt, xtt};
		if(switch_xt_xtt == true)
		{
			factorLbls[0] = xtt;
			factorLbls[1] = xt;
		}
		return factor(factorLbls) + leftMessage + rightMessage;
	}

template<class GM, class ACC>
	double ParallelDiffusion<GM,ACC>::getValueOfUnaryFactor(IndexType variableId, LabelType xt){

		double unaryFactorValue = 0;

		for (auto nthFactorOfVariable = 0; nthFactorOfVariable < _gm.numberOfFactors(variableId); ++nthFactorOfVariable)
		{
			IndexType factorId = _gm.factorOfVariable(variableId,nthFactorOfVariable);
			if (_gm.numberOfVariables(factorId) == 1)
			{
				ValueType unaryFactorValue = _gm[factorId](xt);
				break; // we only expect ONE! unary factor for all variables.
			}
		}

		return unaryFactorValue;
	}

/*
 *
 */
template<class GM, class ACC>
	void ParallelDiffusion<GM,ACC>::diffusionIterationNew(bool onBlackNodes) {

	std::pair<IndexType, IndexType> partitionBds = this->computePartitionBounds();
	//std::cout << _mpi_myRank << "computing [" << partitionBds.first << "," << partitionBds.second << "]" << std::endl;

	// update At
	for (IndexType myVariableIndex=partitionBds.first; myVariableIndex<partitionBds.second; myVariableIndex++) //or i<gm_.numberOfVariables()
	{
		if(this->blackAndWhite(myVariableIndex) != onBlackNodes) {
			continue;
		}

		// compute At
		UnaryFactor at = this->computeAt(myVariableIndex);

		UnaryFactor got2 = _nAt[myVariableIndex];
		_nAt[myVariableIndex] = at;

	}
#ifdef USEMPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif

	// update phi_tt'
	for (IndexType myVariableIndex=partitionBds.first; myVariableIndex<partitionBds.second; myVariableIndex++) //or i<gm_.numberOfVariables()
	{
		// iterate over neighbours
		for (IndexType localFactorIdx = 0; localFactorIdx < _gm.numberOfFactors(myVariableIndex); localFactorIdx++)
		{
			IndexType factorIndex = _gm.factorOfVariable(myVariableIndex,localFactorIdx);
			if (_gm.numberOfVariables(factorIndex) == 1)
			{
				continue;
			}

			for (IndexType localLblId = 0; localLblId < _gm.numberOfLabels(myVariableIndex); localLblId++)
			{

			}



		}


		// update phi

		// distribute phi
	}

	// for all variables t'

		// for all labels x_t \in X_t

			double At_xt = 0;

			// for all neighbouring nodes t'

				// At(xt) = \sum_{t' \in Nt' } min_{x_t' \in X_t'} g_tt'^phi(x_t,x_t') + g_t(x_t)

			// endfor

		// endfor


		// for all neighbouring nodes t'

			// for all labels x_t \in X_t

				// phi_tt -= min x_t' gtt'^phi (x_t, x_t') - w_tt' A t(xt)

			// endfor

		// endfor

	// endfor

}

/*
 * do one iteration of the diffusion algorithm
 */
template<class GM, class ACC>
void ParallelDiffusion<GM,ACC>::diffusionIteration(bool isBlack)
{  
	std::pair<IndexType, IndexType> partitionBds = this->computePartitionBounds(); 
	//std::cout << _mpi_myRank << "computing [" << partitionBds.first << "," << partitionBds.second << "]" << std::endl;
	//for (IndexType myVariableIndex=partitionBds.first + checkerboardOffset; myVariableIndex<partitionBds.second; myVariableIndex+=2) //or i<gm_.numberOfVariables()  


	for (IndexType myVariableIndex=partitionBds.first; myVariableIndex<partitionBds.second; myVariableIndex++) //or i<gm_.numberOfVariables()
	{
		if(this->blackAndWhite(myVariableIndex) != isBlack) {
			continue;
		}
		//VecFactorIndices factorsOfVariable;
		//factorsOfVariable = _variableToFactors[myVariableIndex];
		for (auto factorId=0;factorId<_gm.numberOfFactors(myVariableIndex);++factorId)
			//for(IndexType factIdx : factorsOfVariable)
		{
			IndexType factIdx = _gm.factorOfVariable(myVariableIndex,factorId);

			/*if(_gm[factIdx].variableIndex(0) != myVariableIndex)
	{
	  continue;
	}*/

			// only care about pairwise potentials
			if (_gm[factIdx].numberOfVariables() == 1)
			{
				continue;
			}


			/*
			 *
			 * UPDATE PHI
			 *
			 */
			UnaryFactor& uf = _dualVars[myVariableIndex][factIdx];
			uIterator begin = &uf[0];
			std::pair<uIterator, uIterator> labelsOfDualVariable = std::make_pair(begin, begin+uf.size());
			//// TODO MPI: following line writes to _dualVars and changes its relaxed labeling
			this->computePhi(factIdx,myVariableIndex,labelsOfDualVariable.first,labelsOfDualVariable.second); 

			std::stringstream ss;
			ss << "[";
			for(int l = 0; l < uf.size(); ++l) {
				ss << uf[l] << " ";
			}
			ss << "]";
////			std::cout << "variable " << myVariableIndex << "(" << this->blackAndWhite(myVariableIndex) << ") updated " << factIdx << ": " << ss.str() << std::endl;

			/*
			 * 
			 * DISTRIBUTE UPDATE
			 * 
			 * 
			 */
			// 1. create neighbour data structures for current factor
			FactorType factor = _gm[factIdx];
			std::vector<IndexType> neighboursOfDualUpdates;
			for(int j = 0; j < factor.numberOfVariables(); j++) {
				if(factor.variableIndex(j) == myVariableIndex) {
					continue;
				}

				neighboursOfDualUpdates.push_back(factor.variableIndex(j));
			}

			// 2. send variables being connected to the current factor
			for(int recepient : neighboursOfDualUpdates) {
#ifdef USEMPI
				MPI_Request myreq;
#endif

				// minimize communication overhead:
				if(convertVariableIndexToMPIRank(recepient) == _mpi_myRank) {
					continue;
				}

				std::stringstream ss;
	  ss << "[";
	  for(int l = 0; l < uf.size(); ++l) {
	      ss << uf[l] << " ";
	  }
	  ss << "]";
	  std::cout << "variable " << myVariableIndex << "(black: " << this->blackAndWhite(myVariableIndex) << ") sends " << factIdx << " to " << recepient << "(rank " << convertVariableIndexToMPIRank(recepient) << "). Data: " << ss.str() << std::endl;
#ifdef USEMPI
				MPI_Isend(begin, uf.size(), EMPLOYEDMPIDT, convertVariableIndexToMPIRank(recepient), factIdx+recepient*TAGOFFSET, MPI_COMM_WORLD, &myreq);
				MPI_Request_free(&myreq);
#endif
			}

		}
	}
}

/*
 * do one iteration of the diffusion algorithm
 */
template<class GM, class ACC>
void ParallelDiffusion<GM,ACC>::updateAllStates()
{
#ifdef USEMPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif
	std::pair<IndexType, IndexType> partitionBds = this->computePartitionBounds();
	for (IndexType i=partitionBds.first; i<partitionBds.second; i++) //or i<gm_.numberOfVariables()
	{
		this->updateState(i);
	}
}

/*template<class GM, class ACC>
void ParallelDiffusion<GM,ACC>::computePhi(IndexType factorIndex, IndexType varIndex, uIterator begin, uIterator end)
{
	auto firstVarId = _gm[factorIndex].variableIndex(0);
	auto secondVarId = _gm[factorIndex].variableIndex(1);
	// dirty hack :)
	auto labelId = 0;
	if(secondVarId == varIndex) {
		labelId = 1;
	}
	IndexType secondLabelId;
	labelId == 0 ? secondLabelId = 1 : secondLabelId = 0;
	std::vector<LabelType> labels(2);
	labels[labelId] = 0;
	std::vector<LabelType> label(1);
	label[0] = 0;
	for (auto it = begin; it!= end; ++it)
	{
		labels[secondLabelId]=0;
		// compute minimal factor value and subtract it from it (it = relaxed labels)
		auto mini = getFactorValue(factorIndex, varIndex, labels.begin());
		for (auto i=1; i<_gm.numberOfLabels(secondVarId); ++i)
		{
			labels[secondLabelId] = i;
			auto temp = getFactorValue(factorIndex,varIndex, labels.begin());

			if ( temp < mini )
				mini = temp;
		}
		*it -= mini;
		++labels[labelId];

		*it += _weights[varIndex] * getVariableValue(varIndex, label[0]);

		++label[0];
	}

	//if(varIndex == 4800)
	//  std::cout << "   !" << varIndex << "@" << _mpi_myRank << ";" << factorIndex << ";" << *begin << ";" << *(begin+3) << std::endl;
}
*/

//TODO: readonly for single variable on _dualvars.
//Compute minmal g_{t}^{phi} as current labeling
template<class GM, class ACC>
void ParallelDiffusion<GM, ACC>::updateState(IndexType varIndex)
{
	auto mini = std::numeric_limits< ValueType >::max();
	LabelType label=0, tempLabel;
	std::vector<LabelType> labels(1);
	ValueType temp;
	for (auto i=0; i<_gm.numberOfLabels(varIndex); ++i)
	{
		labels[0] = i;
		temp = this->getVariableValue(varIndex,labels[0]);
		tempLabel = label;
		label = i;
		temp < mini ? mini = temp : label = tempLabel;
	}
	_gPhis[varIndex] = mini;
	_states[varIndex] = label;
}

template<class GM, class ACC>
std::pair<typename GM::IndexType, typename GM::IndexType> ParallelDiffusion<GM, ACC>::computePartitionBounds()
{
	int noVariables = _gm.numberOfVariables();
#ifndef USEMPI
	return std::pair<IndexType, IndexType>(0,noVariables);
#endif
	int partitionSz = noVariables / _mpi_commSize;
	IndexType startIdx = _mpi_myRank * partitionSz;
	IndexType endIdx = (_mpi_myRank+1) * partitionSz;
	if(_mpi_myRank == (_mpi_commSize-1)) { // last index gets potentially smaller chunk
		endIdx = noVariables;
	}
#ifdef DEBUGOUTPUT
	std::cout << "Partition sizes " << _mpi_commSize << " Global Size: " << noVariables << " Partition Size: " << partitionSz << " -> " << " (" << startIdx << "," << endIdx << ")" << std::endl;
#endif
	return std::pair<IndexType, IndexType>(startIdx, endIdx);
}

template<class GM, class ACC>
std::pair<typename GM::IndexType, typename GM::IndexType> ParallelDiffusion<GM, ACC>::computePartitionBounds(int rank)
{
	int noVariables = _gm.numberOfVariables();
#ifndef USEMPI
	return std::pair<IndexType, IndexType>(0,noVariables);
#endif
	int partitionSz = noVariables / _mpi_commSize;
	IndexType startIdx = rank * partitionSz;
	IndexType endIdx = (rank+1) * partitionSz;
	if(rank == (_mpi_commSize-1)) { // last index gets potentially smaller chunk
		endIdx = noVariables;
	}
#ifdef DEBUGOUTPUT
	std::cout << "[given rank] Partition sizes " << _mpi_commSize << " Global Size: " << noVariables << " Partition Size: " << partitionSz << " -> " << " (" << startIdx << "," << endIdx << ")" << std::endl;
#endif
	return std::pair<IndexType, IndexType>(startIdx, endIdx);
}

template<class GM, class ACC>
typename ParallelDiffusion<GM, ACC>::ValueType ParallelDiffusion<GM, ACC>::computeBound()
{
	double bound = 0;
	double bound_recv = 0;

	std::pair<IndexType, IndexType> partitionBds = this->computePartitionBounds();
	for (IndexType varIndex=partitionBds.first; varIndex<partitionBds.second; ++varIndex) //or i<gm_.numberOfVariables()
	{
		bound += _gPhis[varIndex];

		for(auto factorIndex : this->_variableToFactors[varIndex])
		{
			if ((_gm[factorIndex].numberOfVariables() == 1) || (_gm[factorIndex].variableIndex(0) != varIndex) )
			{
				continue; 
			}

			auto mini = std::numeric_limits<ValueType>::max();
			std::vector<LabelType> labels(2);
			for (auto i=0; i<_gm[factorIndex].numberOfLabels(0); ++i)
			{
				for (auto j=0; j<_gm[factorIndex].numberOfLabels(1); ++j)
				{
					labels[0] = i;
					labels[1] = j;
					auto temp = this->getFactorValue(factorIndex, _gm[factorIndex].variableIndex(0), labels.begin());
					if (temp < mini)
						mini = temp;
				}
			}

			bound += mini;
		}
	}	
#ifdef USEMPI
	MPI_Reduce(&bound, &bound_recv, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
else
		bound_recv = bound;
#endif
	return bound_recv;
}

//template<class GM, class ACC>
//typename ParallelDiffusion<GM, ACC>::ValueType ParallelDiffusion<GM, ACC>::computeBoundMPI()
//{
//#ifdef DEBUGOUTPUT
//	std::cout << " BOUND COMPUTATION " << std::endl;
//#endif
//	std::pair<IndexType, IndexType> partitionBds = this->computePartitionBounds();
//
//	double bound = 0;
//	double bound_recv = 0;
//	for (auto factorIndex=0; factorIndex<_gm.numberOfFactors(); ++factorIndex)
//	{
//		IndexType factorLeftVarIdx = _gm[factorIndex].variableIndex(0);
//
//		if( (factorLeftVarIdx < partitionBds.first) || (factorLeftVarIdx >= partitionBds.second)) {
//			continue;
//		}
//
//		//only g_tt'
//		if (_gm[factorIndex].numberOfVariables() > 1)
//		{
//			//std::cout << "Factorindex: " << factorIndex << std::endl;
//			auto mini = std::numeric_limits<ValueType>::max();
//			std::vector<LabelType> labels(2);
//			for (auto i=0; i<_gm[factorIndex].numberOfLabels(0); ++i)
//			{
//				for (auto j=0; j<_gm[factorIndex].numberOfLabels(1); ++j)
//				{
//					labels[0] = i;
//					labels[1] = j;
//					//std::cout << "<- (" << _gm[factorIndex].variableIndex(0) << "," << factorIndex << ")" << std::endl;
//					auto temp = this->getFactorValue(factorIndex, _gm[factorIndex].variableIndex(0), labels.begin());
//					if (temp < mini)
//						mini = temp;
//				}
//			}
//			bound += mini;
//		}
//	}
//	// \sum_{t} min_{x_t}g^{phi}_{t}(x_t) -> unary factors are current labeling
//	//#pragma omp for reduction(+:bound)
//	for (auto varIndex=partitionBds.first; varIndex<partitionBds.second; ++varIndex)
//	{
//		bound += _gPhis[varIndex];
//	}
//	//std::cout << "bound: " << bound << std::endl;
//	MPI_Reduce(&bound, &bound_recv, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
//	bound_recv = bound;
//	return bound_recv;
//}

/*
 * update labels with current state vector
 */
template<class GM, class ACC>
opengm::InferenceTermination ParallelDiffusion<GM,ACC>::arg(std::vector<LabelType>& labels, const size_t) const
{
	labels = _states;
	return opengm::NORMAL;
}

template<class GM, class ACC>
typename ParallelDiffusion<GM, ACC>::ValueType ParallelDiffusion<GM, ACC>::energy()
{
	return _energy ;
}

template<class GM, class ACC>
typename ParallelDiffusion<GM, ACC>::ValueType ParallelDiffusion<GM, ACC>::bound()
{
	return _bound ;
}

// int main(int argc, char* argv[])
// { 
// 	std::srand(2342);

// 	using ValueType = double;
// 	using IndexType = size_t;
// 	using LabelType = size_t;
// 	using VarIdMapType = std::map<IndexType,IndexType>;

// 	// Parameters
// 	constexpr size_t nx = 256; //width of the grid
// 	constexpr size_t ny = 256; //height of the grid
// 	constexpr auto numberOfLabels = 3;
// 	constexpr auto lambda = 0.1 ; // for the labels
// 	constexpr auto eps = 1e-05;
// 	int noIterations = 1;

// 	if(argc > 2) {
// 		noIterations = atoi(argv[2]); 
// 	}

// 	/*// Setting up the graphical model
//     using LabelSpace = opengm::SimpleDiscreteSpace<IndexType,LabelType>;
//     LabelSpace labelSpace(nx*ny, numberOfLabels);
//     using Model = opengm::GraphicalModel<ValueType, opengm::Adder, opengm::ExplicitFunction<ValueType>, LabelSpace>;
//     Model gm(labelSpace);
//     buildGraphicalModel<double, Model>(gm, nx, ny, numberOfLabels, lambda);*/

// 	using Model = TST::GraphicalModel;
// 	Model gm;
// 	opengm::hdf5::load(gm, argv[1], "gm");

// 	std::vector<LabelType> labeling(gm.numberOfVariables());

// 	int mpi_myRank = 0;
// #ifdef USEMPI
// 	MPI_Init(NULL, NULL);
// 	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_myRank);
// #endif
// 	ParallelDiffusion<Model,opengm::Adder> mpiDiffusion(gm, noIterations, eps, nx, ny);
// 	if(mpi_myRank == 0) {
// 		std::cout << "** Diffusion MPI" << std::endl;
// 	}
// 	mpiDiffusion.infer();
// 	mpiDiffusion.arg(labeling);


// #ifdef USEMPI
// 	MPI_Finalize();
// #endif

// 	//std::cout << std::endl << std::endl;
// 	/*
// 	 * 
// 	 * old implementation
// 	 */
// 	//if(mpi_myRank > 0) {
// 	//	return 0;
// 	//}
// 	std::cout << "------------------------------------------------" << std::endl << "------------------------------------------------" << std::endl;
// 	//ReparametrisationStorageMPI<Model> repaStorage(gm);
// 	//DiffusionMPI<Model,opengm::Adder> seqDiffusion(gm, repaStorage, nx, ny, noIterations, eps, 0);
// 	//std::cout << "** Diffusion SEQ" << std::endl;
// 	//seqDiffusion.infer();
// 	//seqDiffusion.arg(labeling);
// 	std::cout << std::endl << std::endl;
// 	std::cout << "------------------------------------------------" << std::endl << "------------------------------------------------" << std::endl;
// 	std::cout << "Energy Diffusion MPI: " << mpiDiffusion.energy() << " bound: " << mpiDiffusion.bound() << std::endl;
// 	//std::cout << "Energy Diffusion SEQ: " << seqDiffusion.value() << " bound: " << seqDiffusion.bound() << std::endl;

// 	/*
// 	 * toy example
// 	 */
// 	std::vector<LabelType> toyLabels(gm.numberOfVariables());
// 	std::cout << "Energy: with all labels 0: " << gm.evaluate(toyLabels.begin()) << std::endl;
// }
