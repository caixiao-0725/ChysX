#include "timeStep/timeStep.h"

namespace CX_NAMESPACE
{
    void timeStepCPU(DeviceHostVector<CxVec3x4T<CxReal>>& x_tilde,DeviceHostVector<CxVec3x4T<CxReal>>& x, DeviceHostVector<CxVec3x4T<CxReal>>& x_old, CxReal tsParam0,CxReal tsParam1)
    {
        // TODO: Implement CPU time step
        for (int i = 0; i < x_tilde.GetSize(); i++)
        {
            x_tilde.GetHost()[i].v[0] = 2.0f*x.GetHost()[i].v[0] - x_old.GetHost()[i].v[0];
            x_tilde.GetHost()[i].v[1] = 2.0f*x.GetHost()[i].v[1] - x_old.GetHost()[i].v[1];
            x_tilde.GetHost()[i].v[2] = 2.0f*x.GetHost()[i].v[2] - x_old.GetHost()[i].v[2];
            x_tilde.GetHost()[i].v[3] = 2.0f*x.GetHost()[i].v[3] - x_old.GetHost()[i].v[3];
        }
    }

    void timeStep(DeviceHostVector<CxVec3x4T<CxReal>>& x_tilde,DeviceHostVector<CxVec3x4T<CxReal>>& x, DeviceHostVector<CxVec3x4T<CxReal>>& x_old, CxReal tsParam0,CxReal tsParam1,
        bool use_gpu, Stream* stream)
    {
        if (use_gpu)
        {
            // TODO: Implement GPU time step
        }
        else
        {
            timeStepCPU(x_tilde, x, x_old, tsParam0, tsParam1);
        }

    }
}