#include <foundation/mat33.h>
#include <cassert>

CX_USING_NAMESPACE

void mat33_test()
{
    CxMat33 m(CxVec3(1.0f, 2.0f, 3.0f), CxVec3(4.0f, 5.0f, 6.0f), CxVec3(7.0f, 8.0f, 9.0f));
    assert(m[0][0] == 1.0f && m[0][1] == 2.0f && m[0][2] == 3.0f);
    assert(m[1][0] == 4.0f && m[1][1] == 5.0f && m[1][2] == 6.0f);
    assert(m[2][0] == 7.0f && m[2][1] == 8.0f && m[2][2] == 9.0f);
}