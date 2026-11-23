#pragma once

#ifdef ZENO_MULTIPROCESS
#include <cstddef>
#include <string>

void viewDecodeClear();
void viewDecodeAppend(const char *buf, size_t n);
void viewDecodeSetFrameCache(const char *path, int gcmax);
void viewDecodeFinish();
std::string getLatestAetherExecutionReportJson();
std::string getPreviousAetherExecutionReportJson();
#endif
