#ifndef IO_PATH_H
#define IO_PATH_H

#include <string>


inline std::string get_asset_path()
{
    return std::string{ ASSET_PATH } + "/";
}

#endif