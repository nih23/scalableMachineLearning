#ifndef SEQUENTIAL_DIFFUSION_HXX
#define SEQUENTIAL_DIFFUSION_HXX

#include <limits>
#include <ctime>

#include <opengm/inference/inference.hxx>
#include <reparametrization.hxx>

/*
 * 
 * Author: Benjamin Naujoks
 * 
 * */

template<class GM, class ACC>
class SequentialDiffusion : public opengm::Inference<GM, ACC>
{
public:  
    using IndexType = typename GM::IndexType;
    using ValueType = typename GM::ValueType;
    using LabelType = typename GM::LabelType;
    
    SequentialDiffusion(GM&, ReparametrisationStorage<GM>& , IndexType, IndexType, IndexType, ValueType, int);
    ~SequentialDiffusion()
    {
        #pragma acc exit data delete(this)
    }
    #pragma acc declare copyin(name_)
    #pragma acc routine seq
    std::string name() const override;

    const GM& graphicalModel() const override;
    ValueType value() const override;
    ValueType bound() const override;
    void computePhi(IndexType, IndexType, typename ReparametrisationStorage<GM>::uIterator, typename ReparametrisationStorage<GM>::uIterator);
    //void computePhiForward(IndexType, typename ReparametrisationStorage<GM>::uIterator, typename ReparametrisationStorage<GM>::uIterator);
    void computeState(IndexType);
    void computeWeights();
    opengm::InferenceTermination infer() override;
    void traverse(bool, bool);
    IndexType variableIndex(const IndexType, const IndexType);
    void processNode(IndexType);
    opengm::InferenceTermination arg(std::vector<LabelType>&, const size_t=1) const override;
private:
    const GM& gm_;
    ReparametrisationStorage<GM>& repaStorage_;
    //std::vector<std::vector<ValueType> > statesVec_;
    std::vector<LabelType> states_;
    std::vector<ValueType> gPhis_;
    ValueType energy_;
    std::vector<ValueType> weights_;
    int variant_;
    std::string name_;
    size_t maxIterations_;
    ValueType convBound_;
    IndexType gridSizeX_;
    IndexType gridSizeY_;
    ValueType bound_;
};

template<class GM, class ACC>
SequentialDiffusion<GM, ACC>::SequentialDiffusion(GM& gm, ReparametrisationStorage<GM>& repaStorage, IndexType gridSizeX, IndexType gridSizeY, IndexType maxIterations, ValueType convBound, int variant) 
: gm_(gm), repaStorage_(repaStorage) , gPhis_(std::vector<ValueType>(gm_.numberOfVariables())), states_(std::vector<LabelType>(gm_.numberOfVariables())), energy_(static_cast<ValueType>(0)), maxIterations_(maxIterations), convBound_(convBound), variant_(variant), weights_(std::vector<ValueType>(gm_.numberOfVariables())), gridSizeX_(gridSizeX), gridSizeY_(gridSizeY), bound_(0.), name_("Diffusion")
{
    #pragma acc enter data copyin(this)
}

// map node index (x,y) to unique variable index
template<class GM, class ACC>
typename SequentialDiffusion<GM, ACC>::IndexType SequentialDiffusion<GM,ACC>::variableIndex(const IndexType x, const IndexType y)
{
    return x + gridSizeX_*y;
}

template <class GM, class ACC>
void SequentialDiffusion<GM,ACC>::computeWeights()
{
    #pragma omp parallel
    {
        #pragma omp for
        #pragma acc parallel loop present(this) copy(weights_[0:gm_.numberOfVariables()])
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
void SequentialDiffusion<GM, ACC>::computeState(IndexType varIndex)
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

template<class GM, class ACC>
void SequentialDiffusion<GM, ACC>::computePhi(IndexType factorIndex, IndexType varIndex, typename ReparametrisationStorage<GM>::uIterator begin, typename ReparametrisationStorage<GM>::uIterator end)
{
    	  auto firstVarId = gm_[factorIndex].variableIndex(0);
	  auto secondVarId = gm_[factorIndex].variableIndex(1);
	  auto labelId = repaStorage_.localIndex(factorIndex,varIndex);
	  IndexType secondLabelId;
	  labelId == 0 ? secondLabelId = 1 : secondLabelId = 0;
	  //std::vector<LabelType> labels(2);
	  //labels[labelId] = states_[varIndex];
	  
// 	  auto mini = this->repaStorage_.getFactorValue(factorIndex, labels.begin());
// 	  for (auto i=1;i<gm_.numberOfLabels(secondVarId);++i)
// 	  {
// 	      labels[secondLabelId] = i;
// 	      auto temp = this->repaStorage_.getFactorValue(factorIndex, labels.begin());
// 	      if ( temp < mini )
// 		  mini = temp;
// 	  }
	  
// 	  auto mini = std::numeric_limits<ValueType>::max();
//           std::vector<LabelType> labels(2);
//           for (auto i=0;i<gm_[factorIndex].numberOfLabels(0);++i)
//           {
//                for (auto j=0;j<gm_[factorIndex].numberOfLabels(1);++j)
//                {
//                     labels[0] = i;
//                     labels[1] = j;
//                     auto temp = this->repaStorage_.getFactorValue(factorIndex,labels.begin());
//                     if (temp < mini)
//                         mini = temp;
//                }
//           }
	
	  std::vector<LabelType> labels(2);
	  labels[labelId] = 0;
	  std::vector<LabelType> label(1);
	  label[0] = 0;
	  for (auto it = begin;it!= end;++it)
	  {
	      labels[secondLabelId]=0;
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
	      *it += weights_[varIndex]*this->repaStorage_.getVariableValue(varIndex,label[0]);
	      ++label[0];
	  }
// 	std::vector<LabelType> label(1);
// 	label[0] = 0;
//         for (auto it = begin;it!= end;++it)
//         {
//             *it -= mini;
//             
//             *it += weights_[varIndex]*this->repaStorage_.getVariableValue(varIndex,label[0]);
//             ++label[0];
//         }
}

template<class GM, class ACC>
void SequentialDiffusion<GM, ACC>::processNode(IndexType varIndex)
{
                   
    for (auto factor=0;factor< gm_.numberOfFactors(varIndex);++factor)
    {
        auto factorIndex = gm_.factorOfVariable(varIndex,factor);
        if (gm_[factorIndex].numberOfVariables() > 1)
        {
            typename ReparametrisationStorage<GM>::uIterator begin, end;
            std::pair<typename ReparametrisationStorage<GM>::uIterator, typename ReparametrisationStorage<GM>::uIterator> iter = repaStorage_.getIterators(factorIndex, repaStorage_.localIndex(factorIndex, varIndex));
            this->computePhi(factorIndex, varIndex, iter.first, iter.second);
        }
    }
}

template<class GM, class ACC>
void SequentialDiffusion<GM, ACC>::traverse(bool wantState, bool isBlack)
{
    if (!wantState)
    {
        auto start = isBlack ? 0 : 1;
        #pragma omp parallel
        {
            #pragma omp for
            for (auto i=start;i<gridSizeX_*gridSizeY_;i+=2) //or i<gm_.numberOfVariables()
            {
                this->processNode(i);
            }
        }
//       for (auto i=0;i<gridSizeX_*gridSizeY_;++i)
//       {
// 	this->processNode(i);
//       }
    }
    else
    {
        #pragma omp parallel
        {
            #pragma omp for
            for (auto i=0;i<gridSizeX_*gridSizeY_;++i)
            {
                this->computeState(i);
            }
        }
    }
}

template<class GM, class ACC>
opengm::InferenceTermination SequentialDiffusion<GM, ACC>::infer()
{
    //Compute the weights
    this->computeWeights();
    
    //Checkerboard Implementation
    std::ofstream myFile("TestPotts.csv");
    auto j = 0;
    for(size_t i=0;i<this->maxIterations_;++i)
    {
        auto before = std::time(nullptr);
        //black
        this->traverse(false,true);
	//white
        this->traverse(false, false);
	//black forward
	//this->traverse(false,true,false);
	//white forward
	//this->traverse(false,false,false);
	//states
        this->traverse(true, true);
      
        //test
        energy_ = gm_.evaluate(states_.begin());
        auto oldBound = bound_;
        bound_ = this->bound();
	auto now = std::time(nullptr);
	myFile << bound_ << std::endl;
        std::cout << "Zeitdifferenz: " << (now-before) << std::endl;
        std::cout << "Energy: " << energy_ << "with bound: " << bound_ << std::endl;
        if ( std::fabs(oldBound - bound_) < convBound_)
        {
            ++j;
            if ( j== 10)
                break;
        }
    }
    myFile.close();
    return opengm::NORMAL;
}

template<class GM, class ACC>
opengm::InferenceTermination SequentialDiffusion<GM, ACC>::arg(std::vector< LabelType >& labels, const size_t) const
{
    labels = states_;
    return opengm::NORMAL;
}

template<class GM, class ACC>
typename SequentialDiffusion<GM, ACC>::ValueType SequentialDiffusion<GM, ACC>::bound() const
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


template<class GM, class ACC>
typename SequentialDiffusion<GM, ACC>::ValueType SequentialDiffusion<GM, ACC>::value() const
{
    return energy_ ;
}

template<class GM, class ACC>
const GM& SequentialDiffusion<GM, ACC>::graphicalModel() const
{
    return gm_;
}

template<class GM, class ACC>
std::string SequentialDiffusion<GM, ACC>::name() const
{
    return name_;
}




    
#endif //SEQUENTIAL_DIFFUSION_HXX