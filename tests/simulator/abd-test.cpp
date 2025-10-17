#include "abd/abd_simulator.h"

CX_USING_NAMESPACE

void abd_test()
{
    ABDSimulator simulator;
    simulator.add_box(CxVec3T<CxReal>(0,0,0), CxVec3T<CxReal>(1,1,1));
    simulator.run();
}