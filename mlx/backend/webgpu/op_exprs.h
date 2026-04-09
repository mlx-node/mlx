// Copyright 2026 Apple Inc.
//
// Shared WGSL expression builders for unary and binary element-wise ops.
// Used by unary.cpp, binary.cpp, ternary.cpp, and compiled.cpp. For the
// standalone kernels the variable names are fixed ("in_val", "a_val",
// "b_val"). For fused Compiled kernels the caller substitutes the tape's
// per-tmp variable names via the substitute_* helpers.

#pragma once

#include <cctype>
#include <stdexcept>
#include <string>

namespace mlx::core::wgpu {

// Return true if `t` is one of the WGSL float types that support native math.
inline bool is_wgsl_float_type(const std::string& t) {
  return t == "f32" || t == "f16";
}

// ---------------------------------------------------------------------------
// Unary op WGSL expressions.
//
// Returns a WGSL expression string that uses the identifier `in_val` as the
// loaded input value. The returned string may cast to `out_type` where needed
// (e.g. for ops that must run in f32 and be cast back).
//
// Throws std::runtime_error on unknown ops.
// ---------------------------------------------------------------------------
inline std::string get_unary_op_expr(
    const char* name,
    const std::string& in_type,
    const std::string& out_type) {
  std::string n(name);
  bool is_float = is_wgsl_float_type(in_type);

  if (n == "Abs") {
    if (in_type == "u32") {
      return "in_val";
    }
    return "abs(in_val)";
  }

  if (n == "Negative") {
    if (in_type == "u32") {
      return "u32(-i32(in_val))";
    }
    return "-in_val";
  }

  if (n == "Exp") {
    if (is_float) return "exp(in_val)";
    return out_type + "(exp(f32(in_val)))";
  }

  if (n == "Expm1") {
    if (in_type == "f32") return "exp(in_val) - 1.0";
    if (in_type == "f16") return "exp(in_val) - 1.0h";
    return out_type + "(exp(f32(in_val)) - 1.0)";
  }

  if (n == "Log") {
    if (is_float) return "log(in_val)";
    return out_type + "(log(f32(in_val)))";
  }

  if (n == "Log1p") {
    if (in_type == "f32") return "log(1.0 + in_val)";
    if (in_type == "f16") return "log(1.0h + in_val)";
    return out_type + "(log(1.0 + f32(in_val)))";
  }

  if (n == "Sigmoid") {
    if (in_type == "f32") return "1.0 / (1.0 + exp(-in_val))";
    if (in_type == "f16") return "1.0h / (1.0h + exp(-in_val))";
    return out_type + "(1.0 / (1.0 + exp(-f32(in_val))))";
  }

  if (n == "Sign") {
    if (in_type == "u32") {
      return "select(0u, 1u, in_val > 0u)";
    }
    if (in_type == "i32") {
      return "select(select(-1, 1, in_val > 0), 0, in_val == 0)";
    }
    return "sign(in_val)";
  }

  if (n == "Sin") {
    if (is_float) return "sin(in_val)";
    return out_type + "(sin(f32(in_val)))";
  }
  if (n == "Cos") {
    if (is_float) return "cos(in_val)";
    return out_type + "(cos(f32(in_val)))";
  }
  if (n == "Tan") {
    if (is_float) return "tan(in_val)";
    return out_type + "(tan(f32(in_val)))";
  }
  if (n == "ArcSin") {
    if (is_float) return "asin(in_val)";
    return out_type + "(asin(f32(in_val)))";
  }
  if (n == "ArcCos") {
    if (is_float) return "acos(in_val)";
    return out_type + "(acos(f32(in_val)))";
  }
  if (n == "ArcTan") {
    if (is_float) return "atan(in_val)";
    return out_type + "(atan(f32(in_val)))";
  }
  if (n == "Sinh") {
    if (is_float) return "sinh(in_val)";
    return out_type + "(sinh(f32(in_val)))";
  }
  if (n == "Cosh") {
    if (is_float) return "cosh(in_val)";
    return out_type + "(cosh(f32(in_val)))";
  }
  if (n == "Tanh") {
    if (is_float) return "tanh(in_val)";
    return out_type + "(tanh(f32(in_val)))";
  }
  if (n == "ArcSinh") {
    if (is_float) return "asinh(in_val)";
    return out_type + "(asinh(f32(in_val)))";
  }
  if (n == "ArcCosh") {
    if (is_float) return "acosh(in_val)";
    return out_type + "(acosh(f32(in_val)))";
  }
  if (n == "ArcTanh") {
    if (is_float) return "atanh(in_val)";
    return out_type + "(atanh(f32(in_val)))";
  }

  if (n == "Rsqrt") {
    if (!is_float) return out_type + "(inverseSqrt(f32(in_val)))";
    return "inverseSqrt(in_val)";
  }

  if (n == "Sqrt") {
    if (!is_float) return out_type + "(sqrt(f32(in_val)))";
    return "sqrt(in_val)";
  }

  if (n == "Square") return "in_val * in_val";

  if (n == "Ceil") {
    if (!is_float) return "in_val";
    return "ceil(in_val)";
  }
  if (n == "Floor") {
    if (!is_float) return "in_val";
    return "floor(in_val)";
  }
  if (n == "Round") {
    if (!is_float) return "in_val";
    return "round(in_val)";
  }

  if (n == "Erf") {
    return out_type +
        "(sign(f32(in_val)) * (1.0 - "
        "(0.254829592 * (1.0/(1.0+0.3275911*abs(f32(in_val)))) + "
        "-0.284496736 * pow(1.0/(1.0+0.3275911*abs(f32(in_val))),2.0) + "
        "1.421413741 * pow(1.0/(1.0+0.3275911*abs(f32(in_val))),3.0) + "
        "-1.453152027 * pow(1.0/(1.0+0.3275911*abs(f32(in_val))),4.0) + "
        "1.061405429 * pow(1.0/(1.0+0.3275911*abs(f32(in_val))),5.0)"
        ") * exp(-f32(in_val)*f32(in_val))))";
  }

  if (n == "ErfInv") {
    return out_type +
        "(sign(f32(in_val)) * sqrt(sqrt("
        "pow(2.0/(3.14159265*0.147) + log(1.0 - f32(in_val)*f32(in_val))*0.5, 2.0)"
        " - log(1.0 - f32(in_val)*f32(in_val))/0.147"
        ") - (2.0/(3.14159265*0.147) + log(1.0 - f32(in_val)*f32(in_val))*0.5)))";
  }

  if (n == "LogicalNot") {
    return "select(" + out_type + "(1), " + out_type + "(0), in_val != 0u)";
  }

  if (n == "BitwiseInvert") {
    return "~in_val";
  }

  throw std::runtime_error(
      std::string("[WebGPU op_exprs] Unknown unary op: ") + name);
}

// ---------------------------------------------------------------------------
// Binary op WGSL expressions.
//
// Returns a WGSL expression string that uses the identifiers `a_val` and
// `b_val` as the two loaded input values.
//
// Throws std::runtime_error on unknown ops.
// ---------------------------------------------------------------------------
inline std::string get_binary_op_expr(
    const char* name,
    const std::string& in_type,
    const std::string& out_type) {
  std::string n(name);

  if (n == "Add") return "a_val + b_val";
  if (n == "Subtract") return "a_val - b_val";
  if (n == "Multiply") return "a_val * b_val";
  if (n == "Divide") return "a_val / b_val";

  if (n == "Remainder") {
    if (in_type == "f32" || in_type == "f16") {
      return "a_val - b_val * floor(a_val / b_val)";
    }
    return "a_val % b_val";
  }

  if (n == "Power") {
    if (in_type == "f32" || in_type == "f16") {
      return "pow(a_val, b_val)";
    }
    return out_type + "(pow(f32(a_val), f32(b_val)))";
  }

  if (n == "Equal")
    return "select(" + out_type + "(0), " + out_type + "(1), a_val == b_val)";
  if (n == "NotEqual")
    return "select(" + out_type + "(0), " + out_type + "(1), a_val != b_val)";
  if (n == "Greater")
    return "select(" + out_type + "(0), " + out_type + "(1), a_val > b_val)";
  if (n == "GreaterEqual")
    return "select(" + out_type + "(0), " + out_type + "(1), a_val >= b_val)";
  if (n == "Less")
    return "select(" + out_type + "(0), " + out_type + "(1), a_val < b_val)";
  if (n == "LessEqual")
    return "select(" + out_type + "(0), " + out_type + "(1), a_val <= b_val)";

  if (n == "LogicalAnd") {
    if (in_type == "u32") {
      return "select(" + out_type + "(0), " + out_type +
          "(1), (a_val != 0u) && (b_val != 0u))";
    }
    return "select(" + out_type + "(0), " + out_type +
        "(1), (a_val != " + in_type + "(0)) && (b_val != " + in_type + "(0)))";
  }
  if (n == "LogicalOr") {
    if (in_type == "u32") {
      return "select(" + out_type + "(0), " + out_type +
          "(1), (a_val != 0u) || (b_val != 0u))";
    }
    return "select(" + out_type + "(0), " + out_type +
        "(1), (a_val != " + in_type + "(0)) || (b_val != " + in_type + "(0)))";
  }

  if (n == "Maximum") return "max(a_val, b_val)";
  if (n == "Minimum") return "min(a_val, b_val)";

  if (n == "LogAddExp") {
    return out_type +
        "(max(f32(a_val), f32(b_val)) + log(1.0 + exp(-abs(f32(a_val) - f32(b_val)))))";
  }

  if (n == "ArcTan2") {
    return out_type + "(atan2(f32(a_val), f32(b_val)))";
  }

  throw std::runtime_error(
      std::string("[WebGPU op_exprs] Unknown binary op: ") + name);
}

// ---------------------------------------------------------------------------
// Substitute tokens in a WGSL expression.
//
// Used by the fused Compiled kernel to replace "in_val" / "a_val" / "b_val"
// with per-tape tmp variable names like "tmp_A" / "tmp_tape3".
//
// This is a plain whole-identifier textual replacement. It's safe because the
// expressions we feed in are short and only contain these tokens as bare
// identifiers (never as substrings of other identifiers — WGSL builtins like
// `atan2`, `floor`, `abs` don't contain "in_val" or "a_val"/"b_val").
// ---------------------------------------------------------------------------
inline std::string replace_all(
    std::string s,
    const std::string& from,
    const std::string& to) {
  if (from.empty()) return s;
  size_t pos = 0;
  while ((pos = s.find(from, pos)) != std::string::npos) {
    // Require identifier boundaries: the character before and after the
    // match must not be an identifier character.
    bool ok_before = (pos == 0) ||
        !(std::isalnum(static_cast<unsigned char>(s[pos - 1])) ||
          s[pos - 1] == '_');
    size_t end = pos + from.size();
    bool ok_after = (end >= s.size()) ||
        !(std::isalnum(static_cast<unsigned char>(s[end])) || s[end] == '_');
    if (ok_before && ok_after) {
      s.replace(pos, from.size(), to);
      pos += to.size();
    } else {
      pos += from.size();
    }
  }
  return s;
}

} // namespace mlx::core::wgpu
