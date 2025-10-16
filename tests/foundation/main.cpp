#include <stdlib.h>

extern void vec2_add_test();
extern void vec3_add_test();
extern void vec4_add_test();
extern void mat33_test();

int main() {

    vec2_add_test();
    vec3_add_test();
    vec4_add_test();
    mat33_test();
    return 0;
}