#ifndef SIMULATOR_BASE_H
#define SIMULATOR_BASE_H

#include "macros.h"
#include "foundation/vec3.h"
#include <string>

namespace CX_NAMESPACE
{

class SimulatorBase {
    public:
        int end_frame = 120;
        CxReal frame_dt = (CxReal)1. / 24;
        CxReal suggested_dt = frame_dt;
        int sub_step = 0;
        CxReal time_elapsed = 0;
        std::string output_directory = "simulation_outputs";
        bool line_search = true;
        bool use_gpu = false;
    
        virtual void initialize(){};
        virtual void calculate_dt(){};
        virtual void restart_prepare(){};
        virtual void advance(CxReal dt) = 0;
        virtual void dump_output(int frame_num) = 0;
    
        virtual void run() = 0;
        
    };
}

#endif // SIMULATOR_BASE_H