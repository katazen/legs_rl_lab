#pragma once

#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>
// #include "os_support.h"

#define __LOG(LEVEL, fmt, ...)                                             \
  nav::get_logger()->log(                                                  \
      spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, LEVEL, fmt, \
      ##__VA_ARGS__)
#define SPD_TRACE(fmt, ...) __LOG(spdlog::level::trace, fmt, ##__VA_ARGS__)
#define SPD_DEBUG(fmt, ...) __LOG(spdlog::level::debug, fmt, ##__VA_ARGS__)
#define SPD_INFO(fmt, ...) __LOG(spdlog::level::info, fmt, ##__VA_ARGS__)
#define SPD_WARN(fmt, ...) __LOG(spdlog::level::warn, fmt, ##__VA_ARGS__)
#define SPD_ERROR(fmt, ...) __LOG(spdlog::level::err, fmt, ##__VA_ARGS__)
#define SPD_CRITICAL(fmt, ...) \
  __LOG(spdlog::level::critical, fmt, ##__VA_ARGS__)

#define TS(name) auto t_##name = std::chrono::steady_clock::now()
#define TE(name)                                             \
  SPD_DEBUG("# Timer_{}: {}ms", #name,                       \
            std::chrono::duration<double, std::milli>(       \
                std::chrono::steady_clock::now() - t_##name) \
                .count())
namespace nav
{
std::shared_ptr<spdlog::logger> get_logger(int file_size = 10);
}  // namespace nav
