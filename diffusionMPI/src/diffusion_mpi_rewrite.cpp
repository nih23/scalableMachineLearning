#include "../header/diffusion_mpi_rewrite.hxx"
//#include "../header/diffusionMPI.hxx"
//#include "parallelDiffusion.cpp"
//#include "diffusionMPI.cpp"
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/graphicalmodel/space/simplediscretespace.hxx>
#include <iostream>

#include <mpi.h>

#define USEMPI
//#define VERBOSE
#ifdef USEMPI
	#include <mpi.h>
#endif

template<class GM, class ACC>
std::string Diffusion_MPI_Rewrite<GM, ACC>::name() const
{
	return _name;
}

/*
 *
 * initialize data structures
 *
 */
template<class GM, class ACC>
 Diffusion_MPI_Rewrite<GM,ACC>::Diffusion_MPI_Rewrite(GM& gm, size_t maxIterations, ValueType convBound, int nx, int ny) : _gm(gm) {
 	_name = "Diffusion MPI";
 	_griddx = nx;
 	_griddy = ny;
 	MPI_Comm_rank(MPI_COMM_WORLD, &_mpi_myRank);
 	MPI_Comm_size(MPI_COMM_WORLD, &_mpi_commSize);


	// project primal factor space into variable space for easy distribution by mpi
 	for (IndexType factorIndex=0; factorIndex<_gm.numberOfFactors(); ++factorIndex)
 	{
 		const typename GM::FactorType& factor = _gm[factorIndex];
		IndexType noLabels = factor.numberOfLabels(0); // left variable
		IndexType noVariables = factor.numberOfVariables();
		IndexType leftVar = factor.variableIndex(0);
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
			_lastColumnId = i+1;
			break;
		}
	}
	//std::cout << "LCid: " << _lastColumnId << std::endl;
}


template<class GM, class ACC>
const GM& Diffusion_MPI_Rewrite<GM,ACC>::graphicalModel() const
{
	return _gm;
}

template <class GM, class ACC>
void Diffusion_MPI_Rewrite<GM,ACC>::computeWeights()
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
 opengm::InferenceTermination Diffusion_MPI_Rewrite<GM,ACC>::infer()
 { 
	// initialize weight vectors
 	this->computeWeights();

 	for(size_t i=0;i<this->_maxIterations;++i)
 	{
 		auto before = std::time(nullptr);
		

		// black fields
		#ifdef VERBOSE
		if(_mpi_myRank == 0)
			std::cout << "1. update dual variables (black fields) R" << _mpi_myRank << std::endl;//}
		#endif
		this->diffusionIteration(true);
		this->sendMPIUpdatesOfDualVariables(true);
		this->receiveMPIUpdatesOfDualVariablesBOOST(true);


		// white fields		
		#ifdef VERBOSE
		if(_mpi_myRank == 0)
			std::cout << "2. update dual variables (white fields) R" << _mpi_myRank << std::endl;//}
		#endif
		this->diffusionIteration(false); 
		this->sendMPIUpdatesOfDualVariables(false);
		this->receiveMPIUpdatesOfDualVariablesBOOST(false);		

		if(_mpi_myRank == 0)
		{
			#ifdef VERBOSE
			if(_mpi_myRank == 0)
				std::cout << "3. update state vector R" << _mpi_myRank << std::endl;//} 
			#endif
			this->updateAllStates();
			auto now = std::time(nullptr);

			_bound = this->computeBound(); // optimize this function as well! since it slows down computational time remarkedly
			now = std::time(nullptr);

			//_energy = this->computeEnergy();
			_energy = this->computeEnergySerial();
			auto oldBound = _bound;
			if(_mpi_myRank == 0) {
				std::cout << i << " energy " << _energy << " bound: " << _bound << std::endl;
			}
		}
	}
	return opengm::NORMAL;
}

template<class GM, class ACC>
typename GM::ValueType Diffusion_MPI_Rewrite<GM,ACC>::computeEnergy() {
	int sz;
	std::pair<IndexType, IndexType> bds;

	// send local state update to master for energy computation
	if(_mpi_myRank > 0) {
		bds = this->computePartitionBounds();
		IndexType* begin = &(_states[bds.first]); // unsigned long!
		sz = bds.second - bds.first + 1;
		MPI_Request myreq;
		MPI_Send(begin, sz, MPI_UINT64_T, 0, 42+_mpi_myRank, MPI_COMM_WORLD); //changed from MPI_UNIT64_T (13.7.2016)
		return 0;
	}

#ifdef VERBOSE
	std::cout << "[d] receiving state updates from children." << std::endl;
#endif

	for(int i = 1; i < _mpi_commSize; i++) {
		MPI_Status status;
		bds = this->computePartitionBounds(i);
		sz = bds.second - bds.first + 1;
		std::vector<IndexType> rcvBuffer;
		rcvBuffer.resize(sz);
		MPI_Recv(&rcvBuffer[0], sz, MPI_UINT64_T, i, 42+i, MPI_COMM_WORLD, &status);

		for(int k = 0; k < sz; k++)
		{
			_states[bds.first+k] = rcvBuffer[k];
		}
	}

	return _gm.evaluate(_states.begin());
}

template<class GM, class ACC>
typename GM::ValueType Diffusion_MPI_Rewrite<GM,ACC>::computeEnergySerial() {
	return _gm.evaluate(_states.begin());
}

/*
 * do one iteration of the diffusion algorithm
 */
template<class GM, class ACC>
void Diffusion_MPI_Rewrite<GM,ACC>::diffusionIteration(bool isBlack)
{  

 	std::pair<IndexType, IndexType> partitionBds = this->computePartitionBounds(); 
	for (IndexType myVariableIndex=partitionBds.first; myVariableIndex<partitionBds.second; myVariableIndex++)
	{
		if(this->blackAndWhite(myVariableIndex) != isBlack) {
			continue;
		}

		for (auto factorId=0;factorId<_gm.numberOfFactors(myVariableIndex);factorId++)
		{
			// only care about pairwise potentials
			IndexType factIdx = _gm.factorOfVariable(myVariableIndex,factorId);
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
			 this->computePhi(factIdx,myVariableIndex,labelsOfDualVariable.first,labelsOfDualVariable.second);
		}
	}
}

	template<class GM, class ACC>
void Diffusion_MPI_Rewrite<GM,ACC>::computePhi(IndexType factorIndex, IndexType varIndex, uIterator begin, uIterator end)
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

	// update dual variable for each label
	for (auto it = begin; it!= end; ++it)
	{
		labels[secondLabelId]=0;
		// compute minimal factor value and subtract it from it (it = relaxed labels)
		auto mini = this->getFactorValue(factorIndex, varIndex, labels.begin());
		for (auto i=1; i<_gm.numberOfLabels(secondVarId); ++i)
		{
			labels[secondLabelId] = i;
			auto temp = this->getFactorValue(factorIndex,varIndex, labels.begin());
			if ( temp < mini )
				mini = temp;
		}
		*it -= mini;
		++labels[labelId];
		*it += _weights[varIndex] * this->getVariableValue(varIndex, label[0]);
		++label[0];
	}
}

template<class GM, class ACC>
	void Diffusion_MPI_Rewrite<GM,ACC>::sendMPIUpdatesOfDualVariables(bool isBlack)
	{
		auto checkerboardOffset = isBlack ? 0 : 1;
		std::pair<IndexType, IndexType> partitionBds = this->computePartitionBounds(); 
		int myRank = _mpi_world_comm.rank();

		boost::unordered_map<IndexType, MapFactorindexDualvariable> sendBuffer;
		for (IndexType myVariableIndex=partitionBds.first; myVariableIndex<partitionBds.second; myVariableIndex++)
		{
			VecFactorIndices factorsOfVariable;
			factorsOfVariable = _variableToFactors[myVariableIndex];
			for(IndexType factIdx : factorsOfVariable)
			{
				for (int i = 0; i < _mpi_world_comm.size(); i++) 
				{
					if(i == myRank) 
					{
						continue;
					}

					std::vector<ValueType> sndBffr = _dualVars[myVariableIndex][factIdx];
					_mpi_world_comm.send(i,factIdx,sndBffr);
				}

			}
		}



	}

template<class GM, class ACC>
	void Diffusion_MPI_Rewrite<GM,ACC>::receiveMPIUpdatesOfDualVariablesBOOST(bool isBlack)
	{
		int myRank = _mpi_world_comm.rank();

		for (int rank = 0; rank < _mpi_world_comm.size(); rank++) 
		{
			if(rank == myRank) 
			{
				continue;
			}

			std::pair<IndexType, IndexType> partitionBds = this->computePartitionBounds(rank); 

			for (IndexType variableIndex=partitionBds.first; variableIndex<partitionBds.second; variableIndex++)
			{
				VecFactorIndices factorsOfVariable;
				factorsOfVariable = _variableToFactors[variableIndex];
				for(IndexType factIdx : factorsOfVariable)
				{
					std::vector<ValueType> recvBffr;
					_mpi_world_comm.recv(rank,factIdx,recvBffr);
					_dualVars[variableIndex][factIdx] = recvBffr;
				}

			}

		}
	}

/*
template<class GM, class ACC>
		void Diffusion_MPI_Rewrite<GM,ACC>::receiveMPIUpdatesOfDualVariables(bool isBlack)
		{
			auto checkerboardOffset = isBlack ? 0 : 1;
			std::pair<IndexType, IndexType> partitionBds = this->computePartitionBounds(); 
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

			// only care about pairwise potentials
			if (factor.numberOfVariables() == 1)
			{
				continue;
			}

			if(convertVariableIndexToMPIRank(factor.variableIndex(0)) != _mpi_myRank) {
			//	continue;
			}  

			// create neighbour data structures for current factor
			std::vector<IndexType> neighboursOfDualUpdates;

			for(int j = 0; j < factor.numberOfVariables(); j++) 
			{
				if(convertVariableIndexToMPIRank(factor.variableIndex(j)) == _mpi_myRank) {
					continue;
				}  
				neighboursOfDualUpdates.push_back(factor.variableIndex(j));
			}

	  		// we only rcv data starting from the 1.5th iteration
	  		// receive data from neighbours
			for(IndexType neighbourIdx : neighboursOfDualUpdates) 
			{
				std::vector<DualType> rcvBuffer;
				rcvBuffer.resize(_dualVars[neighbourIdx][factIdx].size());

				#ifdef DEBUGSEND
					// std::cout <<  _mpi_myRank << " receiving " << factIdx << " from " << convertVariableIndexToMPIRank(neighbourIdx);
				#endif
				MPI_Status status;	  
				MPI_Recv(&(rcvBuffer[0]), rcvBuffer.size(), MPI_DOUBLE, convertVariableIndexToMPIRank(neighbourIdx), factIdx, MPI_COMM_WORLD, &status);
				
				#ifdef DEBUGSEND
					// std::cout << " finished." << std::endl;
				#endif
				//#ifdef DEBUGSEND
				//	std::stringstream ss;
				//	ss << "[";
				for(int l = 0; l < rcvBuffer.size(); ++l) 
				{
				//		ss << rcvBuffer[l] << " ";
					_dualVars[neighbourIdx][factIdx][l] = rcvBuffer[l];
				}
				//	ss << "]";
					//std::cout << " variable " << myVariableIndex << "(black: " << this->blackAndWhite(myVariableIndex) << ", rank " << _mpi_myRank << ") received " << factIdx << " from " << neighbourIdx << " (black: " << this->blackAndWhite(neighbourIdx) << ", rank" << convertVariableIndexToMPIRank(neighbourIdx) << "). Data: " << ss.str() << std::endl;
				//#endif

			}
		}
	}
}*/

/*
 * do one iteration of the diffusion algorithm
 */
template<class GM, class ACC>
 void Diffusion_MPI_Rewrite<GM,ACC>::updateAllStates() 
 {
 	std::pair<IndexType, IndexType> partitionBds = this->computePartitionBounds();
	//for (IndexType i=partitionBds.first; i<partitionBds.second; i++) //or i<gm_.numberOfVariables()
	for (IndexType i=0; i < _gm.numberOfVariables(); i++)
	{
		this->updateState(i);
	}
}

//Compute minmal g_{t}^{phi} as current labeling
template<class GM, class ACC>
void Diffusion_MPI_Rewrite<GM, ACC>::updateState(IndexType varIndex)
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
std::pair<typename GM::IndexType, typename GM::IndexType> Diffusion_MPI_Rewrite<GM, ACC>::computePartitionBounds()
{
	return computePartitionBounds(_mpi_myRank);
}

template<class GM, class ACC>
std::pair<typename GM::IndexType, typename GM::IndexType> Diffusion_MPI_Rewrite<GM, ACC>::computePartitionBounds(int rank)
{
	int noVariables = _gm.numberOfVariables();
	int partitionSz = noVariables / _mpi_commSize;
	IndexType startIdx = rank * partitionSz;
	IndexType endIdx = (rank+1) * partitionSz;
	if(rank == (_mpi_commSize-1)) { // last index gets potentially smaller chunk
		endIdx = noVariables;
	}
#ifdef VERBOSE
	std::cout << "[given rank] Partition sizes " << _mpi_commSize << " Global Size: " << noVariables << " Partition Size: " << partitionSz << " -> " << " (" << startIdx << "," << endIdx << ")" << std::endl;
#endif
	return std::pair<IndexType, IndexType>(startIdx, endIdx);
}

template<class GM, class ACC>
typename Diffusion_MPI_Rewrite<GM, ACC>::ValueType Diffusion_MPI_Rewrite<GM, ACC>::computeBound()
{
	double bound = 0;
	double bound_recv = 0;

	std::pair<IndexType, IndexType> partitionBds = this->computePartitionBounds();
	//for (IndexType varIndex=partitionBds.first; varIndex<partitionBds.second; ++varIndex) //or i<gm_.numberOfVariables()
	for (IndexType varIndex=0; varIndex < _gm.numberOfVariables(); varIndex++)
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

	//std::cout << "b: " << bound << std::endl;
	//MPI_Reduce(&bound, &bound_recv, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	return bound;
}
/*
template<class GM, class ACC>
typename Diffusion_MPI_Rewrite<GM, ACC>::ValueType Diffusion_MPI_Rewrite<GM, ACC>::computeBoundMPI()
{
#ifdef DEBUGOUTPUT
	std::cout << " BOUND COMPUTATION " << std::endl;
#endif
	std::pair<IndexType, IndexType> partitionBds = this->computePartitionBounds();

	double bound = 0;
	double bound_recv = 0;
	for (auto factorIndex=0; factorIndex<_gm.numberOfFactors(); ++factorIndex)
	{
		IndexType factorLeftVarIdx = _gm[factorIndex].variableIndex(0);

		if( (factorLeftVarIdx < partitionBds.first) || (factorLeftVarIdx >= partitionBds.second)) {
			continue;
		}

		//only care about pairwise potentials g_tt'
		if (_gm[factorIndex].numberOfVariables() == 1) 
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
					//std::cout << "<- (" << _gm[factorIndex].variableIndex(0) << "," << factorIndex << ")" << std::endl;
				auto temp = this->getFactorValue(factorIndex, _gm[factorIndex].variableIndex(0), labels.begin());
				if (temp < mini)
					mini = temp;
			}
		}
		bound += mini;
		
	}
	// \sum_{t} min_{x_t}g^{phi}_{t}(x_t) -> unary factors are current labeling
	//#pragma omp for reduction(+:bound)
	for (auto varIndex=partitionBds.first; varIndex<partitionBds.second; ++varIndex)
	{
		bound += _gPhis[varIndex];
	}
	//std::cout << "bound: " << bound << std::endl;
	MPI_Reduce(&bound, &bound_recv, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	bound_recv = bound;
	return bound_recv;
}
*/

/*
 * update labels with current state vector
 */
template<class GM, class ACC>
 opengm::InferenceTermination Diffusion_MPI_Rewrite<GM,ACC>::arg(std::vector<LabelType>& labels, const size_t) const
 {
 	labels = _states;
 	return opengm::NORMAL;
 }

template<class GM, class ACC>
 typename Diffusion_MPI_Rewrite<GM, ACC>::ValueType Diffusion_MPI_Rewrite<GM, ACC>::energy()
 {
 	return _energy ;
 }

template<class GM, class ACC>
 typename Diffusion_MPI_Rewrite<GM, ACC>::ValueType Diffusion_MPI_Rewrite<GM, ACC>::bound()
 {
 	return _bound ;
 }

 int main(int argc, char* argv[])
 { 
 	std::srand(2342);

 	using ValueType = double;
 	using IndexType = size_t;
 	//using LabelType = size_t;
 	using LabelType = size_t;
 	using VarIdMapType = std::map<IndexType,IndexType>;

	// Parameters
	constexpr size_t nx = 256; //width of the grid
	constexpr size_t ny = 256; //height of the grid
	constexpr auto eps = 1e-05;
	int noIterations = 1;

	if(argc > 2) {
		noIterations = atoi(argv[2]); 
	}

	using Model = TST::GraphicalModel;
	Model gm;
	opengm::hdf5::load(gm, argv[1], "gm");

	std::vector<LabelType> labeling(gm.numberOfVariables());

	// initialize communication framework
	int mpi_myRank = 0;
	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_myRank);
	Diffusion_MPI_Rewrite<Model,opengm::Adder> mpiDiffusion(gm, noIterations, eps, nx, ny);
	if(mpi_myRank == 0) {
		std::cout << "** Diffusion MPI" << std::endl;
	}

	// infer labelling
	mpiDiffusion.infer();
	mpiDiffusion.arg(labeling);

	// shutdown mpi
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();

	 if(mpi_myRank > 0) {
	 	return 0; 
	 }
	 std::cout << "------------------------------------------------" << std::endl << "------------------------------------------------" << std::endl;

	 return 0;
	}
