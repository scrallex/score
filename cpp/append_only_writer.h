#pragma once

#include <cstdio>
#include <stdexcept>
#include <string>

#include "manifold_record.h"

class AppendOnlyWriter {
  std::FILE* file_{nullptr};

public:
  explicit AppendOnlyWriter(const std::string& path) {
    file_ = std::fopen(path.c_str(), "ab");
    if (!file_) {
      throw std::runtime_error("Failed to open manifold log for append");
    }
  }

  ~AppendOnlyWriter() {
    if (file_) {
      std::fclose(file_);
    }
  }

  AppendOnlyWriter(const AppendOnlyWriter&) = delete;
  AppendOnlyWriter& operator=(const AppendOnlyWriter&) = delete;

  void append(const ManifoldRecord& record) {
    if (std::fwrite(&record, sizeof(record), 1, file_) != 1) {
      throw std::runtime_error("Failed to append manifold record");
    }
  }

  void flush() {
    if (file_ && std::fflush(file_) != 0) {
      throw std::runtime_error("Failed to flush manifold log");
    }
  }
};
