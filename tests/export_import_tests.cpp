// Copyright © 2024 Apple Inc.

#include <atomic>
#include <filesystem>
#include <stdexcept>
#include <thread>
#include <vector>

#include "doctest/doctest.h"

#include "mlx/export.h"
#include "mlx/fast_primitives.h"
#include "mlx/mlx.h"

using namespace mlx::core;

namespace {
std::string get_temp_file(const std::string& name) {
  return std::filesystem::temp_directory_path().append(name).string();
}
} // namespace

TEST_CASE("test export basic functions") {
  std::string file_path = get_temp_file("model.mlxfn");

  auto fun = [](std::vector<array> x) -> std::vector<array> {
    return {negative(exp(x[0]))};
  };

  export_function(file_path, fun, {array({1.0, 2.0})});

  auto imported_fun = import_function(file_path);

  // Check num inputs mismatch throws
  CHECK_THROWS_AS(
      imported_fun({array({1.0}), array({2.0})}), std::invalid_argument);

  // Check shape mismatch throws
  CHECK_THROWS_AS(imported_fun({array({1.0})}), std::invalid_argument);

  // Check type mismatch throws
  CHECK_THROWS_AS(imported_fun({array({1.0}, float16)}), std::invalid_argument);

  auto expected = fun({array({1.0, -1.0})});
  auto out = imported_fun({array({1.0, -1.0})});
  CHECK(allclose(expected[0], out[0]).item<bool>());
}

TEST_CASE("test export function with no inputs") {
  auto fun = [](std::vector<array> x) -> std::vector<array> {
    return {zeros({2, 2})};
  };

  std::string file_path = get_temp_file("model.mlxfn");

  export_function(file_path, fun, {});

  auto imported_fun = import_function(file_path);

  auto expected = fun({});
  auto out = imported_fun({});
  CHECK(allclose(expected[0], out[0]).item<bool>());
}

TEST_CASE("test export multi output primitives") {
  std::string file_path = get_temp_file("model.mlxfn");

  auto fun = [](std::vector<array> x) -> std::vector<array> {
    return {divmod(x[0], x[1])};
  };

  auto inputs = std::vector<array>{array({5.0, -10.0}), array({3.0, -2.0})};
  export_function(file_path, fun, inputs);

  auto imported_fun = import_function(file_path);

  auto expected = fun(inputs);
  auto out = imported_fun(inputs);
  CHECK(allclose(expected[0], out[0]).item<bool>());
  CHECK(allclose(expected[1], out[1]).item<bool>());
}

TEST_CASE("test export primitives with state") {
  std::string file_path = get_temp_file("model.mlxfn");

  auto fun = [](std::vector<array> x) -> std::vector<array> {
    return {argpartition(x[0], 2, 0)};
  };

  auto x = array({1, 3, 2, 4, 5, 7, 6, 8}, {4, 2});
  export_function(file_path, fun, {x});

  auto imported_fun = import_function(file_path);

  auto expected = fun({x});
  auto out = imported_fun({x});
  CHECK(allclose(expected[0], out[0]).item<bool>());
}

TEST_CASE("test export functions with kwargs") {
  std::string file_path = get_temp_file("model.mlxfn");

  auto fun = [](const Kwargs& kwargs) -> std::vector<array> {
    return {kwargs.at("x") + kwargs.at("y")};
  };

  export_function(file_path, fun, {{"x", array(1)}, {"y", array(2)}});
  auto fn = import_function(file_path);

  // Must use kwargs
  CHECK_THROWS(fn({array(1), array(2)}));

  // Wrong number of keys
  CHECK_THROWS(fn({{"x", array(1)}, {"y", array(2)}, {"z", array(3)}}));

  // Wrong keys
  CHECK_THROWS(fn({{"a", array(1)}, {"b", array(2)}}));

  // Works
  auto out = fn({{"x", array(1)}, {"y", array(2)}})[0];
  CHECK_EQ(out.item<int>(), 3);
  out = fn({}, {{"x", array(1)}, {"y", array(2)}})[0];
  CHECK_EQ(out.item<int>(), 3);
}

TEST_CASE("test export function with variable inputs") {
  std::string file_path = get_temp_file("model.mlxfn");

  auto fun = [](const std::vector<array>& args) -> std::vector<array> {
    auto out = array({1, 1, 1, 1});
    for (auto x : args) {
      out = out + x;
    }
    return {out};
  };

  {
    auto fn_exporter = exporter(file_path, fun);
    fn_exporter({array(0), array(0)});
    fn_exporter({array(0), array(0), array(0)});
  }

  auto imported_fun = import_function(file_path);

  // Call with two inputs
  auto out = imported_fun({array(1), array(2)})[0];

  CHECK(array_equal(out, array({4, 4, 4, 4})).item<bool>());

  // Call with three inputs
  out = imported_fun({array(1), array(2), array(3)})[0];
  CHECK(array_equal(out, array({7, 7, 7, 7})).item<bool>());
}

TEST_CASE("test export function on different stream") {
  std::string file_path = get_temp_file("model.mlxfn");

  auto fun = [](const std::vector<array>& args) -> std::vector<array> {
    return {abs(args[0], Stream(1000, Device::cpu))};
  };

  export_function(file_path, fun, {array({0, 1, 2})});

  // Should make a new stream that we can run computation on
  eval(import_function(file_path)({array({0, 1, 2})}));
}

TEST_CASE("test export import with metadata") {
  std::string file_path = get_temp_file("model.mlxfn");

  auto fun = [](const std::vector<array>& args) -> std::vector<array> {
    return {abs(args[0])};
  };

  std::string metadata = "{\"name\": \"model\", \"params\": 7000000000}";

  export_function(file_path, fun, {array({0, 1, 2})}, false, metadata);

  auto imported = import_function(file_path);
  CHECK(imported.metadata() == metadata);
  eval(imported({array({0, 1, 2})}));

  // No metadata gives an empty string
  export_function(file_path, fun, {array({0, 1, 2})});
  CHECK(import_function(file_path).metadata().empty());

  // Metadata survives the per-trace header rewrite of a multi-trace export
  {
    auto fn_exporter = exporter(file_path, fun, false, metadata);
    fn_exporter({array({0, 1})});
    fn_exporter({array({0, 1, 2})});
  }
  CHECK(import_function(file_path).metadata() == metadata);
}

TEST_CASE("test export import custom kernel shared across threads") {
  if (!metal::is_available()) {
    return;
  }
  std::string file_path = get_temp_file("custom_kernel.mlxfn");

  // The padding keeps every thread inside the key computation at once;
  // without it the concurrent first calls below rarely overlap.
  std::string source =
      "uint elem = thread_position_in_grid.x;\n"
      "out[elem] = a[elem] + 1;\n"
      "// " +
      std::string(1 << 18, 'x') + "\n";
  auto kernel = fast::metal_kernel("shared_import", {"a"}, {"out"}, source);
  auto fun = [&kernel](const std::vector<array>& args) -> std::vector<array> {
    auto& a = args[0];
    int n = static_cast<int>(a.size());
    return kernel(
        {a},
        {a.shape()},
        {a.dtype()},
        {n, 1, 1},
        {n, 1, 1},
        {},
        std::nullopt,
        false,
        Device::gpu);
  };
  auto a = arange(8);
  export_function(file_path, fun, {a});

  // One deserialized primitive serves every call of the imported function
  auto imported = import_function(file_path);
  auto out = imported({a});
  auto* custom = dynamic_cast<fast::CustomKernel*>(&out[0].primitive());
  REQUIRE(custom != nullptr);

  constexpr int num_threads = 16;
  constexpr int calls_per_thread = 64;
  std::atomic<bool> start{false};
  std::atomic<int> moved{0};
  std::vector<const char*> first_data(num_threads, nullptr);
  std::vector<std::string> values(num_threads);
  std::vector<std::thread> threads;
  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([&, t]() {
      while (!start.load(std::memory_order_acquire)) {
      }
      const std::string& key = custom->library_name();
      first_data[t] = key.data();
      values[t] = key;
      for (int i = 1; i < calls_per_thread; ++i) {
        if (custom->library_name().data() != first_data[t]) {
          moved++;
        }
      }
    });
  }
  start.store(true, std::memory_order_release);
  for (auto& th : threads) {
    th.join();
  }

  CHECK_EQ(moved.load(), 0);
  CHECK_FALSE(values[0].empty());
  for (int t = 1; t < num_threads; ++t) {
    CHECK_EQ(first_data[t], first_data[0]);
    CHECK_EQ(values[t], values[0]);
  }
  CHECK_EQ(custom->library_name().data(), first_data[0]);

  // The fallback key names a library that runs
  CHECK(array_equal(out[0], fun({a})[0]).item<bool>());
}
