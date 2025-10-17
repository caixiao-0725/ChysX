#include <vec2.h>
#include <vec3.h>
#include <vec4.h>
#include <cassert>
#include <iostream>

CX_USING_NAMESPACE

void vec2_add_test()
{
    CxVec2 a(1.0f, 2.0f);
    CxVec2 b(3.0f, 4.0f);
    CxVec2 c = a + b;
    assert(c.x == 4.0f && c.y == 6.0f);
}

void vec3_add_test()
{
    CxVec3 a(1.0f, 2.0f, 3.0f);
    CxVec3 b(4.0f, 5.0f, 6.0f);
    CxVec3 c = a + b;
    assert(c.x == 5.0f && c.y == 7.0f && c.z == 9.0f);
}

void vec4_add_test()
{
    CxVec4 a(1.0f, 2.0f, 3.0f, 4.0f);
    CxVec4 b(5.0f, 6.0f, 7.0f, 8.0f);
    CxVec4 c = a + b;
    assert(c.x == 6.0f && c.y == 8.0f && c.z == 10.0f && c.w == 12.0f);
}
