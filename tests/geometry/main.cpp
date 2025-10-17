#include <iostream>

extern void triangle_mesh_cube_test();

int main() {
    std::cout << "Running Geometry Tests..." << std::endl;
    std::cout << "=========================" << std::endl;
    
    triangle_mesh_cube_test();
    
    std::cout << "All geometry tests completed successfully!" << std::endl;
    return 0;
}