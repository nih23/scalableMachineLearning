#include "../header/diffusionMPI.hxx"
#include <fstream>


#include <cstdlib>
#include <vector>
#include <cmath>
#include <limits>
#include <numeric>
#include <algorithm>
#include <ctime>


#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/graphicalmodel/space/simplediscretespace.hxx>

//#define DEBUGOUTPUT
#define DEBUGVARIDX 1000
#define DEBUGINTERESTINGFACTORIDX 67023

// map node index (x,y) to unique variable index
size_t variableIndex(const size_t x, const size_t y, const size_t nx)
{
	return x + nx*y;
}

/*
 * 
 * 
 * BUILD GRAPHICAL MODEL
 * 
 * 
 */
template<class T, class Model>
void buildGraphicalModel(Model& gm, const size_t nx, const size_t ny, const size_t numberOfLabels, const T lambda)
{    
	// g_t(x_labels) for every node (x,y)
	const size_t shape[] = {numberOfLabels};
	for (auto y=0;y < ny; ++y)
	{
		for (auto x=0;x < nx; ++x)
		{
			//function
			opengm::ExplicitFunction<T> f(shape,shape+1);
			for (auto s=0;s<numberOfLabels;++s)
			{
				f(s) = ( (1.0-lambda)* (T)std::rand() ) /(T)RAND_MAX ;
				//f(s) = double(s+1) / double(numberOfLabels);
				//f(s) = 0;
			}
			typename Model::FunctionIdentifier g_t = gm.addFunction(f);

			//factor
			size_t varIndices[] = {variableIndex(x,y,nx)};
			gm.addFactor(g_t, varIndices, varIndices+1);
		}
	}

	//add one 2nd order function
	const size_t shape2[] = {numberOfLabels, numberOfLabels};
	opengm::ExplicitFunction<T> f2(shape2, shape2+2);

	// g_tt'(x_labels,x'_labels) for each pair of nodes (x1,y1), (x2,y2) which are adjacent on the grid
	for (auto label1=0;label1<numberOfLabels;++label1)
	{
		for (auto label2=0;label2<numberOfLabels;++label2)
		{
			f2(label1, label2) = ((T)std::rand() / (T)RAND_MAX);
			//f2(label1, label2) = (double(label1+1) + double(label2+1)) / double(2*(numberOfLabels));

		}
	}
	typename Model::FunctionIdentifier g_tt = gm.addFunction(f2);
	for (auto y=0;y < ny; ++y)
	{
		for (auto x=0;x < nx; ++x)
		{
			if (x+1<nx) // (x,y) -- (x+1,y)
			{
				size_t varIndices[] = {variableIndex(x,y,nx), variableIndex(x+1,y,nx)};
				std::sort(varIndices, varIndices +2);
				gm.addFactor(g_tt, varIndices, varIndices+2);
			}
			if (y+1<ny) // (x,y) -- (x,y+1)
			{
				size_t varIndices[] = {variableIndex(x,y,nx), variableIndex(x,y+1,nx)};
				std::sort(varIndices, varIndices +2);
				gm.addFactor(g_tt, varIndices, varIndices+2);
			}
		}
	}
}


template<class GM, class ACC>
DiffusionMPI<GM, ACC>::DiffusionMPI(GM& gm, ReparametrisationStorageMPI<GM>& repaStorage, IndexType gridSizeX, IndexType gridSizeY, IndexType maxIterations, ValueType convBound, int variant) 
: gm_(gm), repaStorage_(repaStorage) , gPhis_(std::vector<ValueType>(gm_.numberOfVariables())), states_(std::vector<LabelType>(gm_.numberOfVariables())), energy_(static_cast<ValueType>(0)), maxIterations_(maxIterations), convBound_(convBound), variant_(variant), weights_(std::vector<ValueType>(gm_.numberOfVariables())), gridSizeX_(gridSizeX), gridSizeY_(gridSizeY), bound_(0.), name_("Diffusion")
  {
#pragma acc enter data copyin(this)
	//TODO: do MPI initialization here
  }

// map node index (x,y) to unique variable index
template<class GM, class ACC>
typename DiffusionMPI<GM, ACC>::IndexType DiffusionMPI<GM,ACC>::variableIndex(const IndexType x, const IndexType y)
{
	return x + gridSizeX_*y;
}



template<class GM, class ACC>
opengm::InferenceTermination DiffusionMPI<GM, ACC>::arg(std::vector< LabelType >& labels, const size_t) const
{
	labels = states_;
	return opengm::NORMAL;
}

template<class GM, class ACC>
typename DiffusionMPI<GM, ACC>::ValueType DiffusionMPI<GM, ACC>::value() const
{
	return energy_ ;
}

template<class GM, class ACC>
const GM& DiffusionMPI<GM, ACC>::graphicalModel() const
{
	return gm_;
}

template<class GM, class ACC>
std::string DiffusionMPI<GM, ACC>::name() const
{
	return name_;
}

/*
 * *************************
 * BEGIN MPI PARALLELIZATION
 * *************************
 * 
 */

template<class GM, class ACC>
opengm::InferenceTermination DiffusionMPI<GM, ACC>::infer()
{
	//TODO: init MPI with number of nodes

	//Compute the weights
	this->computeWeights();

	//Checkerboard Implementation
	std::ofstream myFile("TestPotts.csv");
	auto j = 0;
	//this->maxIterations_ = 20;
	for(size_t i=0;i<this->maxIterations_;++i)
	{
		auto before = std::time(nullptr);
		//std::cout << "diffusion iteration" << std::endl;
		//black
		//std::cout << "1. black" << std::endl;
		this->traverse(false,true);

		//white
		//std::cout << "2. white" << std::endl;
		this->traverse(false, false);

		//states
		//std::cout << "3. states" << std::endl;
		this->traverse(true, true);

		auto now = std::time(nullptr);
		//std::cout << "dt (phi) " << (now-before) << std::endl;
		now = std::time(nullptr);
		bound_ = this->bound(); // optimize this function as well! slows down computation remarkedly
		energy_ = gm_.evaluate(states_.begin());
		auto oldBound = bound_;
		bound_ = this->bound();

		//myFile << bound_ << std::endl;

		//std::cout << "dt (bound) " << (std::time(nullptr)-now) << std::endl;
		std::cout << i << " energy " << energy_ << " bound: " << bound_ << std::endl;
		/*if ( std::fabs(oldBound - bound_) < convBound_)
        {
            ++j;
            if ( j== 10)
                break;
        }*/
	}
	myFile.close();
	return opengm::NORMAL;
}

template<class GM, class ACC>
void DiffusionMPI<GM, ACC>::traverse(bool wantState, bool isBlack)
{
	//TODO: partition loops according to mpi process grid

	if (!wantState)
	{
		auto start = isBlack ? 0 : 1;
		for (auto i=start;i<gm_.numberOfVariables();i+=2) //or i<gm_.numberOfVariables()
		{
			this->processVariable(i);
		}

	}
	else
	{
		for (auto i=0;i<gm_.numberOfVariables();++i)
		{
			this->computeState(i);
		}
	}
}

template<class GM, class ACC>
void DiffusionMPI<GM, ACC>::processVariable(IndexType varIndex)
{
	for (auto factor=0;factor< gm_.numberOfFactors(varIndex);++factor)
	{
		auto factorIndex = gm_.factorOfVariable(varIndex,factor);
		if (gm_[factorIndex].numberOfVariables() > 1)
		{
			typename ReparametrisationStorageMPI<GM>::uIterator begin, end;
			std::pair<typename ReparametrisationStorageMPI<GM>::uIterator, typename ReparametrisationStorageMPI<GM>::uIterator> iter = repaStorage_.getIterators(factorIndex, repaStorage_.localIndex(factorIndex, varIndex));
			begin = iter.first;
			end = iter.second;
			std::stringstream ss_pre;
			ss_pre << "[";
			for (auto it = begin;it!= end;++it)
			{
				ss_pre << *it << " ";
			}
			ss_pre << "]";


			this->computePhi(factorIndex, varIndex, iter.first, iter.second);

			std::stringstream ss;
			ss << "[";
			for (auto it = begin;it!= end;++it)
			{
				ss << *it << " ";
			}
			ss << "]";

////			std::cout << "variable " << varIndex << " updated " << factorIndex << ". Data: " << ss_pre.str() << " -> " << ss.str() <<  std::endl;
		}
	}
}

// test
template<class GM, class ACC>
void DiffusionMPI<GM, ACC>::computePhi(IndexType factorIndex, IndexType varIndex, typename ReparametrisationStorageMPI<GM>::uIterator begin, typename ReparametrisationStorageMPI<GM>::uIterator end)
{
	auto firstVarId = gm_[factorIndex].variableIndex(0);
	auto secondVarId = gm_[factorIndex].variableIndex(1);
	auto labelId = repaStorage_.localIndex(factorIndex,varIndex);
	IndexType secondLabelId;
	labelId == 0 ? secondLabelId = 1 : secondLabelId = 0;
	std::vector<LabelType> labels(2);
	labels[labelId] = 0;
	std::vector<LabelType> label(1);
	label[0] = 0;
	for (auto it = begin;it!= end;++it)
	{
		// if(varIndex == 4800)
		// std::cout << "it " << *it << ";";
		labels[secondLabelId]=0;
		// compute minimal factor value and subtract it from it (it = ?)
		auto mini = this->repaStorage_.getFactorValue(factorIndex, labels.begin());
		for (auto i=1;i<gm_.numberOfLabels(secondVarId);++i)
		{
			labels[secondLabelId] = i;
			auto temp = this->repaStorage_.getFactorValue(factorIndex, labels.begin());

			if ( temp < mini )
				mini = temp;

		}
		*it -= mini;
		++labels[labelId];
		//if(varIndex == 4800)
			//std::cout << mini  << ";" << label[0] << ";;;";
		//if( varIndex == 10998 && factorIndex == 33225 ) {
		//std::cout << "oooo PHI DEBUG SER: l=" << label[0] << " - gVV=" << this->repaStorage_.getVariableValue(varIndex, label[0]) << std::endl;
		//}
		*it += weights_[varIndex] * this->repaStorage_.getVariableValue(varIndex,label[0]);
		//	*it += weights_[varIndex] * factorIndex;

		++label[0];

	}

	//if(varIndex == 4800)
	// std::cout << "  !" << varIndex << ";" << factorIndex << ";" << *begin << ";" << *(begin+3) <<  std::endl;
}

template<class GM, class ACC>
void DiffusionMPI<GM,ACC>::outputDualvars() {
	/*int noVars = gm_.numberOfVariables();
  for(IndexType fi = 0; i < gm_.numberOfFactors; i++) {
	Factor f = gm_[i];
	IndexType varIdx = f.variableIndex(0);
	UnaryFactor uf = _dualVars[varIdx][f];
	  for(ValueType uf_value : uf) {
	      std::cout << uf_value << " ; ";
	  }
	  std::cout << std::endl;
	}

  }*/
	/*      std::cout  << i << ";" ;
      auto dv_i = _dualVars[i]; 
      for(UnaryFactor fi : dv_i) {
	for(ValueType vi : fi) {
	  std::cout << vi << ";";
	}
      }
      std::cout << std::endl;
  }*/
}

template <class GM, class ACC>
void DiffusionMPI<GM,ACC>::computeWeights()
{
	{
		for (auto varIndex=0;varIndex<gm_.numberOfVariables();++varIndex)
		{
			switch (variant_)
			{
			case 0:
				weights_[varIndex] = 1./((ValueType)gm_.numberOfFactors(varIndex)); // w_0 = 0
				break;
			case 1:
				weights_[varIndex] = 1./((ValueType)gm_.numberOfFactors(varIndex)+1); //w_o = w_tt
				break;
			}
		}
	}
}

//Compute minmal g_{t}^{phi} as current labeling
template<class GM, class ACC>
void DiffusionMPI<GM, ACC>::computeState(IndexType varIndex)
{
	auto mini = std::numeric_limits< ValueType >::max();
	LabelType label=0, tempLabel;
	std::vector<LabelType> labels(1);
	ValueType temp;
	for (auto i=0;i<gm_.numberOfLabels(varIndex);++i)
	{
		labels[0] = i;
		temp = this->repaStorage_.getVariableValue(varIndex,labels[0]);
		tempLabel = label;
		label = i;
		temp < mini ? mini = temp : label = tempLabel;

	}
	gPhis_[varIndex] = mini;
	states_[varIndex] = label;
}



/*
 * 
 * BOUND COMPUTATION
 * TODO: MPI ME!
 */
template<class GM, class ACC>
typename DiffusionMPI<GM, ACC>::ValueType DiffusionMPI<GM, ACC>::bound() const
{
	auto bound = (ValueType)0;
	// \sum_{tt'} min_{x_t,x_t'}g^{phi}_{tt'}(x_t,x_t')
#pragma omp parallel
	{
#pragma omp for schedule(dynamic) reduction(+:bound)
		for (auto factorIndex=0;factorIndex<gm_.numberOfFactors();++factorIndex)
		{
			//only g_tt'
			if (gm_[factorIndex].numberOfVariables() > 1)
			{
				auto mini = std::numeric_limits<ValueType>::max();
				std::vector<LabelType> labels(2);
				for (auto i=0;i<gm_[factorIndex].numberOfLabels(0);++i)
				{
					for (auto j=0;j<gm_[factorIndex].numberOfLabels(1);++j)
					{
						labels[0] = i;
						labels[1] = j;
						auto temp = this->repaStorage_.getFactorValue(factorIndex,labels.begin());
						if (temp < mini)
							mini = temp;
					}
				}
				bound += mini;
			}
		}
		// \sum_{t} min_{x_t}g^{phi}_{t}(x_t) -> unary factors are current labeling
#pragma omp for reduction(+:bound)
		for (auto varIndex=0;varIndex<gm_.numberOfVariables();++varIndex)
		{
			bound += gPhis_[varIndex];
		}

	}
	return bound;
}




/*
 * 
 * 
 *  MAIN
 * 
 * 
 * 
 */

/*int main_old()
{
	using ValueType = double;
	using IndexType = size_t;
	using LabelType = size_t;
	using VarIdMapType = std::map<IndexType,IndexType>;
	// Parameters
	constexpr size_t nx = 256; //width of the grid
	constexpr size_t ny = 256; //height of the grid
	constexpr auto numberOfLabels = 3;
	constexpr auto lambda = 0.1 ; // for the labels
	constexpr auto eps = 1e-05;
	constexpr auto N = 100; //Max iterations
	constexpr auto variant = 0;

	// Setting up the graphical model
	using LabelSpace = opengm::SimpleDiscreteSpace<IndexType,LabelType>;
	LabelSpace labelSpace(nx*ny, numberOfLabels);
	using Model = opengm::GraphicalModel<ValueType, opengm::Adder, opengm::ExplicitFunction<ValueType>, LabelSpace>;
	Model gm(labelSpace);
	buildGraphicalModel<double, Model>(gm, nx, ny, numberOfLabels, lambda);
	ReparametrisationStorageMPI<Model> repaStorage(gm);
	DiffusionMPI<Model,opengm::Adder> seqDiffusion(gm, repaStorage, nx, ny, N, eps, 0);



	//auto factors[] = gm.getVariableValue(1);



	seqDiffusion.infer();

	std::vector<LabelType> labeling(gm.numberOfVariables());
	seqDiffusion.arg(labeling);

	/*size_t variableIndex = 0;
    for(size_t y = 0; y < ny; ++y) 
    {
       for(size_t x = 0; x < nx; ++x) 
       {
          std::cout << labeling[variableIndex] << ' ';
          ++variableIndex;
       }   
       std::cout << std::endl;
   }
	 
	std::cout << "Energy: " << seqDiffusion.value() << std::endl;
	std::vector<LabelType> toyLabels(gm.numberOfVariables());
	std::cout << "Energy: with all lables 0: " << gm.evaluate(toyLabels.begin()) << std::endl;

}*/
