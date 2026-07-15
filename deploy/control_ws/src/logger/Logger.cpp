#include "Logger.h"

#include "spdlog/async.h"
#include "spdlog/sinks/rotating_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"

namespace nav
{
std::shared_ptr<spdlog::logger> get_logger(int file_size)
{
    static std::shared_ptr<spdlog::logger> logger = nullptr;
    static auto thread_pool =
        std::make_shared<spdlog::details::thread_pool>(8192, 1);
    if (logger != nullptr) return logger;
    static std::once_flag flag;
    std::call_once(flag, [&]
    {
        std::vector<spdlog::sink_ptr> sinks;
        sinks.push_back(std::make_shared <
                        spdlog::sinks::ansicolor_stdout_sink_mt > ()); // console
        // logger
        // std::string fileLog = "/data/cache/log/task_executor_runner_log.log";
        std::string fileLog = "./log/workspace_server.log";
        sinks.push_back(std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
                            fileLog, 1024 * 1024 * 2, file_size));         // file logger
        sinks[0]->set_level(spdlog::level::debug);  // log level for console logger
        logger = std::make_shared<spdlog::async_logger>("", sinks.begin(), sinks.end(), thread_pool);
        // logger->set_pattern("[%^%L %m-%d %H:%M:%S.%e] %n %v%$ %s:%#");
        logger->set_pattern("[%^%L %m-%d %H:%M:%S.%e %6t] %n %v%$ %s:%#");
        logger->set_level(spdlog::level::debug);
    });
    return logger;
}
}  // namespace nav
