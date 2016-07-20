#include <iostream>
//#include <mpi.h>
#include "opengm/inference/inference.hxx"
#include "opengm/inference/movemaker.hxx"
#include "opengm/inference/visitors/visitors.hxx"

#include <limits>
#include <ctime>

#include "reparametrization_mpi.hxx"


template<class GM, class ACC>
class DiffusionMPI : public opengm::Inference<GM, ACC>
{
public:  
    using IndexType = typename GM::IndexType;
    using ValueType = typename GM::ValueType;
    using LabelType = typename GM::LabelType;
    
    DiffusionMPI(GM&, ReparametrisationStorageMPI<GM>& , IndexType, IndexType, IndexType, ValueType, int);
    ~DiffusionMPI()
    {
        #pragma acc exit data delete(this)
    }
    #pragma acc declare copyin(name_)
    #pragma acc routine seq
    std::string name() const override;

    const GM& graphicalModel() const override;
    ValueType value() const override;
    ValueType bound() const override;
    void computePhi(IndexType, IndexType, typename ReparametrisationStorageMPI<GM>::uIterator, typename ReparametrisationStorageMPI<GM>::uIterator);
    //void computePhiForward(IndexType, typename ReparametrisationStorage<GM>::uIterator, typename ReparametrisationStorage<GM>::uIterator);
    void computeState(IndexType);
    void computeWeights();
    opengm::InferenceTermination infer() override;
    void traverse(bool, bool);
    IndexType variableIndex(const IndexType, const IndexType);
    void processVariable(IndexType);
    opengm::InferenceTermination arg(std::vector<LabelType>&, const size_t=1) const override;
    
private:
      void outputDualvars();
    const GM& gm_;
    ReparametrisationStorageMPI<GM>& repaStorage_;
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