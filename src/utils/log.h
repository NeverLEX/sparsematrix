#pragma once
#include <iostream>
#include <sstream>
#include <pthread.h>

#define LOG_ERR \
    logger::MessageLogger(logger::LogLevel::ERROR, __func__).stream()
#define LOG_WARN \
    logger::MessageLogger(logger::LogLevel::WARNING, __func__).stream()
#define LOG_INFO \
    logger::MessageLogger(logger::LogLevel::INFO, __func__).stream()

#if ((defined DEBUG) || (defined _DEBUG))
#define LOG_DBG \
    logger::MessageLogger(logger::LogLevel::DBG, __func__).stream()
#else
#define LOG_DBG \
    if (true) {} \
    else logger::MessageLogger(logger::LogLevel::DBG, __func__).stream()
#endif

namespace logger {

enum LogLevel {
    ERROR       = 0,
    WARNING     = 1,
    INFO        = 2,
    DBG         = 3
};

class LoggerMutex {
public:
    LoggerMutex() { pthread_mutex_init(&mutex_, nullptr); }
    ~LoggerMutex() { pthread_mutex_destroy(&mutex_); }
    void Lock() { pthread_mutex_lock(&mutex_); }
    void Unlock() { pthread_mutex_unlock(&mutex_); }

private:
    pthread_mutex_t mutex_;
};

static LoggerMutex g_logger_mutex;

class MessageLogger {
public:
    MessageLogger(LogLevel lv, const char *func) {
        log_msg_.lv = lv;
        log_msg_.func = func;
    }
    ~MessageLogger() {
        std::stringstream header;
        if (LogLevel::ERROR == log_msg_.lv) {
            header << "(E) ";
        } else if (LogLevel::WARNING == log_msg_.lv) {
            header << "(W) ";
        } else if (LogLevel::INFO == log_msg_.lv) {
            header << "(I) ";
        } else if (LogLevel::DBG == log_msg_.lv) {
            header << "(D) ";
        } else {
            header << "(U) ";
        }
        if (nullptr == log_msg_.func) log_msg_.func = "unknow func";
        header << log_msg_.func << ": ";
        g_logger_mutex.Lock();
        std::cout << header.str() << ss_.str() << std::endl;
        g_logger_mutex.Unlock();
    }
    inline std::ostream &stream() { return ss_; }
private:
    struct LogMessage {
        LogLevel lv;
        const char *func;
    };
    LogMessage log_msg_;
    std::ostringstream ss_;
};

}  // logger
