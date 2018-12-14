#pragma once
#include <iostream>

#define _TO_STRIN_(x)                       #x
#define _TO_STRING(x)                       _TO_STRIN_(x)

#define _MACRO_COMBIN_(x,y)                 x##y
#define _MACRO_COMBINE(x,y)                 _MACRO_COMBIN_(x,y)

// define DEBUG or _DEBUG  enable diagnosis
#if ((defined DEBUG) || (defined _DEBUG))
#  define _ENABLE_DIAGNOSIS  1
#else
#  define _ENABLE_DIAGNOSIS  0
#endif

// timer
#if (_ENABLE_DIAGNOSIS)
#  define add_time_probe(name)          utility::Timer _g_counter_##name
#  define start_clock(name)             { /*extern add_time_probe(name);*/ _g_counter_##name.start(); }
#  define stop_clock(name)              { /*extern add_time_probe(name);*/ _g_counter_##name.stop(); }
#  define pause_clock(name)             { /*extern add_time_probe(name);*/ _g_counter_##name.pause(); }
#  define reset_clock(name)             { /*extern add_time_probe(name);*/ _g_counter_##name.reset(); }
#  define total_elapsed_ms(name)        _g_counter_##name.total_ms()
#else
#  define add_time_probe(name)
#  define start_clock(name)
#  define pause_clock(name)
#  define reset_clock(name)
#  define total_elapsed_ms(name)
#endif

// __runtime_verify_code__ for runtime and __compile_verify_code__ for compile time
#if (_ENABLE_DIAGNOSIS)
#  define __diagnosis_code__(...)           __VA_ARGS__
#  define __runtime_verify_code__(name,c)   static const int _MACRO_COMBINE(__rt_check_,__COUNTER__)=((c)? 0: ((std::cout << "Failed to verify `"#name"'.\nThe expression is false:\n"#c"\nFile: "__FILE__"("_TO_STRING(__LINE__)")\n"), exit(-1), 0));
#  define __compile_verify_code__(name,c)   static const int _MACRO_COMBINE(__ct_check_,__COUNTER__)[(c)?1:-1]={0};
#else
#  define __diagnosis_code__(...)
#  define __runtime_verify_code__(name,c)
#  define __compile_verify_code__(name,c)
#endif

#include "log.h"
#include "utility.h"
