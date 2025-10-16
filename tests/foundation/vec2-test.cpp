#include <vec2.h>
#include <cassert>
#include <iostream>

CX_USING_NAMESPACE

void vec2_add_test()
{
    CxVec2 a(1.0f, 2.0f);
    CxVec2 b(3.0f, 4.0f);
    CxVec2 c = a + b;
    printf("%f  %f \n", c.x, c.y);
    assert(c.x == 4.0f && c.y == 6.0f);
}