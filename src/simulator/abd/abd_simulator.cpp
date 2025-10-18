#include "abd/abd_simulator.h"
#include "foundation/io_path.h"

namespace CX_NAMESPACE
{
    void ABDSimulator::initialize()
    {
    }

    void ABDSimulator::calculate_dt()
    {
       
    }

    void ABDSimulator::restart_prepare()
    {
        
    }

    void ABDSimulator::advance(CxReal dt)
    {
        dt *= 10;
    }

    void ABDSimulator::dump_output(int /*frame_num*/)
    {
        // Stub: hook for writing meshes/frames
        // No-op until an output format is defined
    }

    void ABDSimulator::run()
    {

        for (int frame = 0; frame < end_frame; ++frame)
        {
            advance(suggested_dt);
        }
    }

    void ABDSimulator::add_box(const CxVec3T<CxReal>& center, const CxVec3T<CxReal>& size)
    {
        m_model.emplace_back(center, size);
        auto &model = m_model.back();
        std::string cubePath = get_asset_path() + "cube/cube.obj";
        model.ReadFromObjFile(cubePath);
        model.initialize();
    }

}