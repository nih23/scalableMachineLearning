#include <iostream>
#include "../header/diffusionMPI.hxx"

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/graphicalmodel/space/simplediscretespace.hxx>



// map node index (x,y) to unique variable index
size_t variableIndex(const size_t x, const size_t y, const size_t nx)
{
    return x + nx*y;
}






