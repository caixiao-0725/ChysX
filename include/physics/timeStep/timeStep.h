#ifndef TIME_STEP_CUH
#define TIME_STEP_CUH

#include "foundation/vec3.h"
#include "foundation/macros.h"
#include "cudaAPI/context.h"
#include "cudaAPI/stream.h"
#include "cudaAPI/device_host_vector.h"

namespace CX_NAMESPACE
{
    void timeStep(DeviceHostVector<CxVec3x4T<CxReal>>& x_tilde,DeviceHostVector<CxVec3x4T<CxReal>>& x, DeviceHostVector<CxVec3x4T<CxReal>>& x_old, 
        CxReal tsParam0,CxReal tsParam1,
        bool use_gpu = false, Stream* stream = nullptr);
}
#endif