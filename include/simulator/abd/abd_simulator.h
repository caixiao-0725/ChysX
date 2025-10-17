#ifndef ABD_SIMULATOR_H
#define ABD_SIMULATOR_H

#include "simulator_base.h"
#include "abd_model.h"
#include <vector>

namespace CX_NAMESPACE
{
    class ABDSimulator : public SimulatorBase
    {
    public:
        ABDSimulator() {};
        ~ABDSimulator() {};

        void initialize() override;
        void calculate_dt() override;
        void restart_prepare() override;
        void advance(CxReal dt) override;
        void dump_output(int frame_num) override;
        void run() override;

        void add_box(const CxVec3T<CxReal>& center, const CxVec3T<CxReal>& size);

    private:
        std::vector<ABDModel> m_model;
    };
}
#endif // ABD_SIMULATOR_H