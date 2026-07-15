#pragma once
#include <string>
#include <cstdio>
#include <cstdint>
#include <ctime>
#include <chrono>
#include <stdlib.h>
#include "logger/Logger.h"

// windows
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <direct.h>
// linux
#elif __linux__ || __APPLE__
#include <sys/stat.h>
// others
#else
#error "OS not supported!"
#endif

namespace os
{
using namespace std;

// exists
inline bool exists(const string& str)
{
#ifdef _WIN32
    return GetFileAttributesW(str.c_str()) != INVALID_FILE_ATTRIBUTES;
#else
    struct stat sb;
    return stat(str.c_str(), &sb) == 0;
#endif
}

// filesize
inline int getFileSize(const string & filename)
{
    FILE * pfile = fopen(filename.c_str(), "rb") ;
    if(!pfile)
    {
        printf("file: open [%s] failed!, errno:%d\n", filename.c_str(), errno) ;
        SPD_ERROR("File: open [{}] failed!, errno: {}", filename, errno);
        return 0 ;
    }

    int size = 0 ;
    if(fseek(pfile, 0, SEEK_END) == 0)
    {
        size = (int)ftell(pfile) ;
    }
    fclose(pfile);
    return size ;
}

// mkdir
inline void mkdir(const string& str)
{
#ifdef _WIN32
    _mkdir(str.c_str());
#else
    ::mkdir(str.c_str(), 0777);
#endif
};

// get localtime and convert to string
// http://www.cplusplus.com/reference/ctime/strftime/
inline string getLocaltime(const char* str)
{
    auto tNow = chrono::system_clock::now();
    time_t rawtime = chrono::system_clock::to_time_t(tNow);

#ifdef _WIN32
    std::tm timeinfo;
    localtime_s(&timeinfo, &rawtime);
#else
    std::tm timeinfo;
    localtime_r(&rawtime, &timeinfo);
#endif

    char buffer [80];
    strftime(buffer, 80, str, &timeinfo);

    return string(buffer);
};

// to nanoseconds
template <class _Clock>
inline uint64_t time_pointToTimestamp(std::chrono::time_point<_Clock> now)
{
    auto e = now.time_since_epoch();
    uint64_t ts = std::chrono::duration_cast<std::chrono::nanoseconds>(e).count();
    return ts;
}

// get time stamp
inline uint64_t now()
{
    return time_pointToTimestamp(std::chrono::steady_clock::now());
}
inline int getThreadID()
{
#ifdef SYS_gettid
    int tid = syscall(SYS_gettid) ;
#else
    int tid = 0 ;
#endif
    return tid ;
}


}
