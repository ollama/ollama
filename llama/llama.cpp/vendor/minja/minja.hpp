/*
    Copyright 2024 Google LLC

    Use of this source code is governed by an MIT-style
    license that can be found in the LICENSE file or at
    https://opensource.org/licenses/MIT.
*/
// SPDX-License-Identifier: MIT
#pragma once

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <exception>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <regex>
#include <sstream>
#include <string>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

namespace minja {

class Context;

struct Options {
    bool trim_blocks;  // removes the first newline after a block
    bool lstrip_blocks;  // removes leading whitespace on the line of the block
    bool keep_trailing_newline;  // don't remove last newline
};

struct ArgumentsValue;

inline std::string normalize_newlines(const std::string & s) {
#ifdef _WIN32
  static const std::regex nl_regex("\r\n");
  return std::regex_replace(s, nl_regex, "\n");
#else
  return s;
#endif
}

/* Values that behave roughly like in Python. */
class Value {
public:
  using CallableType = std::function<Value(const std::shared_ptr<Context> &, ArgumentsValue &)>;
  using FilterType = std::function<Value(const std::shared_ptr<Context> &, ArgumentsValue &)>;

private:
  using ObjectType = nlohmann::ordered_map<json, Value>;  // Only contains primitive keys
  using ArrayType = std::vector<Value>;

  std::shared_ptr<ArrayType> array_;
  std::shared_ptr<ObjectType> object_;
  std::shared_ptr<CallableType> callable_;
  json primitive_;

  Value(const std::shared_ptr<ArrayType> & array) : array_(array) {}
  Value(const std::shared_ptr<ObjectType> & object) : object_(object) {}
  Value(const std::shared_ptr<CallableType> & callable) : object_(std::make_shared<ObjectType>()), callable_(callable) {}

  /* Python-style string repr */
  static void dump_string(const json & primitive, std::ostringstream & out, char string_quote = '\'') {
    if (!primitive.is_string()) throw std::runtime_error("Value is not a string: " + primitive.dump());
    auto s = primitive.dump();
    if (string_quote == '"' || s.find('\'') != std::string::npos) {
      out << s;
      return;
    }
    // Reuse json dump, just changing string quotes
    out << string_quote;
    for (size_t i = 1, n = s.size() - 1; i < n; ++i) {
      if (s[i] == '\\' && s[i + 1] == '"') {
        out << '"';
        i++;
      } else if (s[i] == string_quote) {
        out << '\\' << string_quote;
      } else {
        out << s[i];
      }
    }
    out << string_quote;
  }
  void dump(std::ostringstream & out, int indent = -1, int level = 0, bool to_json = false) const {
    auto print_indent = [&](int level) {
      if (indent > 0) {
          out << "\n";
          for (int i = 0, n = level * indent; i < n; ++i) out << ' ';
      }
    };
    auto print_sub_sep = [&]() {
      out << ',';
      if (indent < 0) out << ' ';
      else print_indent(level + 1);
    };

    auto string_quote = to_json ? '"' : '\'';

    if (is_null()) out << "null";
    else if (array_) {
      out << "[";
      print_indent(level + 1);
      for (size_t i = 0; i < array_->size(); ++i) {
        if (i) print_sub_sep();
        (*array_)[i].dump(out, indent, level + 1, to_json);
      }
      print_indent(level);
      out << "]";
    } else if (object_) {
      out << "{";
      print_indent(level + 1);
      for (auto begin = object_->begin(), it = begin; it != object_->end(); ++it) {
        if (it != begin) print_sub_sep();
        if (it->first.is_string()) {
          dump_string(it->first, out, string_quote);
        } else {
          out << string_quote << it->first.dump() << string_quote;
        }
        out << ": ";
        it->second.dump(out, indent, level + 1, to_json);
      }
      print_indent(level);
      out << "}";
    } else if (callable_) {
      throw std::runtime_error("Cannot dump callable to JSON");
    } else if (is_boolean() && !to_json) {
      out << (this->to_bool() ? "True" : "False");
    } else if (is_string() && !to_json) {
      dump_string(primitive_, out, string_quote);
    } else {
      out << primitive_.dump();
    }
  }

public:
  Value() {}
  Value(const bool& v) : primitive_(v) {}
  Value(const int64_t & v) : primitive_(v) {}
  Value(const double& v) : primitive_(v) {}
  Value(const std::nullptr_t &) {}
  Value(const std::string & v) : primitive_(v) {}
  Value(const char * v) : primitive_(std::string(v)) {}

  Value(const json & v) {
    if (v.is_object()) {
      auto object = std::make_shared<ObjectType>();
      object->reserve(v.size());
      for (auto it = v.begin(); it != v.end(); ++it) {
        object->emplace_back(it.key(), Value(it.value()));
      }
      object_ = std::move(object);
    } else if (v.is_array()) {
      auto array = std::make_shared<ArrayType>();
      array->reserve(v.size());
      for (const auto& item : v) {
        array->push_back(Value(item));
      }
      array_ = array;
    } else {
      primitive_ = v;
    }
  }

  std::vector<Value> keys() {
    if (!object_) throw std::runtime_error("Value is not an object: " + dump());
    std::vector<Value> res;
    for (const auto& item : *object_) {
      res.push_back(item.first);
    }
    return res;
  }

  size_t size() const {
    if (is_object()) return object_->size();
    if (is_array()) return array_->size();
    if (is_string()) return primitive_.get<std::string>().length();
    throw std::runtime_error("Value is not an array or object: " + dump());
  }

  static Value array(const std::vector<Value> values = {}) {
    auto array = std::make_shared<ArrayType>();
    for (const auto& item : values) {
      array->push_back(item);
    }
    return Value(array);
  }
  static Value object(const std::shared_ptr<ObjectType> object = std::make_shared<ObjectType>()) {
    return Value(object);
  }
  static Value callable(const CallableType & callable) {
    return Value(std::make_shared<CallableType>(callable));
  }

  void insert(size_t index, const Value& v) {
    if (!array_)
      throw std::runtime_error("Value is not an array: " + dump());
    array_->insert(array_->begin() + index, v);
  }
  void push_back(const Value& v) {
    if (!array_)
      throw std::runtime_error("Value is not an array: " + dump());
    array_->push_back(v);
  }
  Value pop(const Value& index) {
    if (is_array()) {
      if (array_->empty())
        throw std::runtime_error("pop from empty list");
      if (index.is_null()) {
        auto ret = array_->back();
        array_->pop_back();
        return ret;
      } else if (!index.is_number_integer()) {
        throw std::runtime_error("pop index must be an integer: " + index.dump());
      } else {
        auto i = index.get<int>();
        if (i < 0 || i >= static_cast<int>(array_->size()))
          throw std::runtime_error("pop index out of range: " + index.dump());
        auto it = array_->begin() + (i < 0 ? array_->size() + i : i);
        auto ret = *it;
        array_->erase(it);
        return ret;
      }
    } else if (is_object()) {
      if (!index.is_hashable())
        throw std::runtime_error("Unhashable type: " + index.dump());
      auto it = object_->find(index.primitive_);
      if (it == object_->end())
        throw std::runtime_error("Key not found: " + index.dump());
      auto ret = it->second;
      object_->erase(it);
      return ret;
    } else {
      throw std::runtime_error("Value is not an array or object: " + dump());
    }
  }
  Value get(const Value& key) {
    if (array_) {
      if (!key.is_number_integer()) {
        return Value();
      }
      auto index = key.get<int>();
      return array_->at(index < 0 ? array_->size() + index : index);
    } else if (object_) {
      if (!key.is_hashable()) throw std::runtime_error("Unhashable type: " + dump());
      auto it = object_->find(key.primitive_);
      if (it == object_->end()) return Value();
      return it->second;
    }
    return Value();
  }
  void set(const Value& key, const Value& value) {
    if (!object_) throw std::runtime_error("Value is not an object: " + dump());
    if (!key.is_hashable()) throw std::runtime_error("Unhashable type: " + dump());
    (*object_)[key.primitive_] = value;
  }
  Value call(const std::shared_ptr<Context> & context, ArgumentsValue & args) const {
    if (!callable_) throw std::runtime_error("Value is not callable: " + dump());
    return (*callable_)(context, args);
  }

  bool is_object() const { return !!object_; }
  bool is_array() const { return !!array_; }
  bool is_callable() const { return !!callable_; }
  bool is_null() const { return !object_ && !array_ && primitive_.is_null() && !callable_; }
  bool is_boolean() const { return primitive_.is_boolean(); }
  bool is_number_integer() const { return primitive_.is_number_integer(); }
  bool is_number_float() const { return primitive_.is_number_float(); }
  bool is_number() const { return primitive_.is_number(); }
  bool is_string() const { return primitive_.is_string(); }
  bool is_iterable() const { return is_array() || is_object() || is_string(); }

  bool is_primitive() const { return !array_ && !object_ && !callable_; }
  bool is_hashable() const { return is_primitive(); }

  bool empty() const {
    if (is_null())
      throw std::runtime_error("Undefined value or reference");
    if (is_string()) return primitive_.empty();
    if (is_array()) return array_->empty();
    if (is_object()) return object_->empty();
    return false;
  }

  void for_each(const std::function<void(Value &)> & callback) const {
    if (is_null())
      throw std::runtime_error("Undefined value or reference");
    if (array_) {
      for (auto& item : *array_) {
        callback(item);
      }
    } else if (object_) {
      for (auto & item : *object_) {
        Value key(item.first);
        callback(key);
      }
    } else if (is_string()) {
      for (char c : primitive_.get<std::string>()) {
        auto val = Value(std::string(1, c));
        callback(val);
      }
    } else {
      throw std::runtime_error("Value is not iterable: " + dump());
    }
  }

  bool to_bool() const {
    if (is_null()) return false;
    if (is_boolean()) return get<bool>();
    if (is_number()) return get<double>() != 0;
    if (is_string()) return !get<std::string>().empty();
    if (is_array()) return !empty();
    return true;
  }

  int64_t to_int() const {
    if (is_null()) return 0;
    if (is_boolean()) return get<bool>() ? 1 : 0;
    if (is_number()) return static_cast<int64_t>(get<double>());
    if (is_string()) {
      try {
        return std::stol(get<std::string>());
      } catch (const std::exception &) {
        return 0;
      }
    }
    return 0;
  }

  bool operator<(const Value & other) const {
    if (is_null())
      throw std::runtime_error("Undefined value or reference");
    if (is_number() && other.is_number()) return get<double>() < other.get<double>();
    if (is_string() && other.is_string()) return get<std::string>() < other.get<std::string>();
    throw std::runtime_error("Cannot compare values: " + dump() + " < " + other.dump());
  }
  bool operator>=(const Value & other) const { return !(*this < other); }

  bool operator>(const Value & other) const {
    if (is_null())
      throw std::runtime_error("Undefined value or reference");
    if (is_number() && other.is_number()) return get<double>() > other.get<double>();
    if (is_string() && other.is_string()) return get<std::string>() > other.get<std::string>();
    throw std::runtime_error("Cannot compare values: " + dump() + " > " + other.dump());
  }
  bool operator<=(const Value & other) const { return !(*this > other); }

  bool operator==(const Value & other) const {
    if (callable_ || other.callable_) {
      if (callable_.get() != other.callable_.get()) return false;
    }
    if (array_) {
      if (!other.array_) return false;
      if (array_->size() != other.array_->size()) return false;
      for (size_t i = 0; i < array_->size(); ++i) {
        if (!(*array_)[i].to_bool() || !(*other.array_)[i].to_bool() || (*array_)[i] != (*other.array_)[i]) return false;
      }
      return true;
    } else if (object_) {
      if (!other.object_) return false;
      if (object_->size() != other.object_->size()) return false;
      for (const auto& item : *object_) {
        if (!item.second.to_bool() || !other.object_->count(item.first) || item.second != other.object_->at(item.first)) return false;
      }
      return true;
    } else {
      return primitive_ == other.primitive_;
    }
  }
  bool operator!=(const Value & other) const { return !(*this == other); }

  bool contains(const char * key) const { return contains(std::string(key)); }
  bool contains(const std::string & key) const {
    if (array_) {
      return false;
    } else if (object_) {
      return object_->find(key) != object_->end();
    } else {
      throw std::runtime_error("contains can only be called on arrays and objects: " + dump());
    }
  }
  bool contains(const Value & value) const {
    if (is_null())
      throw std::runtime_error("Undefined value or reference");
    if (array_) {
      for (const auto& item : *array_) {
        if (item.to_bool() && item == value) return true;
      }
      return false;
    } else if (object_) {
      if (!value.is_hashable()) throw std::runtime_error("Unhashable type: " + value.dump());
      return object_->find(value.primitive_) != object_->end();
    } else {
      throw std::runtime_error("contains can only be called on arrays and objects: " + dump());
    }
  }
  void erase(size_t index) {
    if (!array_) throw std::runtime_error("Value is not an array: " + dump());
    array_->erase(array_->begin() + index);
  }
  void erase(const std::string & key) {
    if (!object_) throw std::runtime_error("Value is not an object: " + dump());
    object_->erase(key);
  }
  const Value& at(const Value & index) const {
    return const_cast<Value*>(this)->at(index);
  }
  Value& at(const Value & index) {
    if (!index.is_hashable()) throw std::runtime_error("Unhashable type: " + dump());
    if (is_array()) return array_->at(index.get<int>());
    if (is_object()) return object_->at(index.primitive_);
    throw std::runtime_error("Value is not an array or object: " + dump());
  }
  const Value& at(size_t index) const {
    return const_cast<Value*>(this)->at(index);
  }
  Value& at(size_t index) {
    if (is_null())
      throw std::runtime_error("Undefined value or reference");
    if (is_array()) return array_->at(index);
    if (is_object()) return object_->at(index);
    throw std::runtime_error("Value is not an array or object: " + dump());
  }

  template <typename T>
  T get(const std::string & key, T default_value) const {
    if (!contains(key)) return default_value;
    return at(key).get<T>();
  }

  template <typename T>
  T get() const {
    if (is_primitive()) return primitive_.get<T>();
    throw std::runtime_error("get<T> not defined for this value type: " + dump());
  }

  std::string dump(int indent=-1, bool to_json=false) const {
    std::ostringstream out;
    dump(out, indent, 0, to_json);
    return out.str();
  }

  Value operator-() const {
      if (is_number_integer())
        return -get<int64_t>();
      else
        return -get<double>();
  }
  std::string to_str() const {
    if (is_string()) return get<std::string>();
    if (is_number_integer()) return std::to_string(get<int64_t>());
    if (is_number_float()) return std::to_string(get<double>());
    if (is_boolean()) return get<bool>() ? "True" : "False";
    if (is_null()) return "None";
    return dump();
  }
  Value operator+(const Value& rhs) const {
      if (is_string() || rhs.is_string()) {
        return to_str() + rhs.to_str();
      } else if (is_number_integer() && rhs.is_number_integer()) {
        return get<int64_t>() + rhs.get<int64_t>();
      } else if (is_array() && rhs.is_array()) {
        auto res = Value::array();
        for (const auto& item : *array_) res.push_back(item);
        for (const auto& item : *rhs.array_) res.push_back(item);
        return res;
      } else {
        return get<double>() + rhs.get<double>();
      }
  }
  Value operator-(const Value& rhs) const {
      if (is_number_integer() && rhs.is_number_integer())
        return get<int64_t>() - rhs.get<int64_t>();
      else
        return get<double>() - rhs.get<double>();
  }
  Value operator*(const Value& rhs) const {
      if (is_string() && rhs.is_number_integer()) {
        std::ostringstream out;
        for (int64_t i = 0, n = rhs.get<int64_t>(); i < n; ++i) {
          out << to_str();
        }
        return out.str();
      }
      else if (is_number_integer() && rhs.is_number_integer())
        return get<int64_t>() * rhs.get<int64_t>();
      else
        return get<double>() * rhs.get<double>();
  }
  Value operator/(const Value& rhs) const {
      if (is_number_integer() && rhs.is_number_integer())
        return get<int64_t>() / rhs.get<int64_t>();
      else
        return get<double>() / rhs.get<double>();
  }
  Value operator%(const Value& rhs) const {
    return get<int64_t>() % rhs.get<int64_t>();
  }
};

struct ArgumentsValue {
  std::vector<Value> args;
  std::vector<std::pair<std::string, Value>> kwargs;

  bool has_named(const std::string & name) {
    for (const auto & p : kwargs) {
      if (p.first == name) return true;
    }
    return false;
  }

  Value get_named(const std::string & name) {
    for (const auto & [key, value] : kwargs) {
      if (key == name) return value;
    }
    return Value();
  }

  bool empty() {
    return args.empty() && kwargs.empty();
  }

  void expectArgs(const std::string & method_name, const std::pair<size_t, size_t> & pos_count, const std::pair<size_t, size_t> & kw_count) {
    if (args.size() < pos_count.first || args.size() > pos_count.second || kwargs.size() < kw_count.first || kwargs.size() > kw_count.second) {
      std::ostringstream out;
      out << method_name << " must have between " << pos_count.first << " and " << pos_count.second << " positional arguments and between " << kw_count.first << " and " << kw_count.second << " keyword arguments";
      throw std::runtime_error(out.str());
    }
  }
};

template <>
inline json Value::get<json>() const {
  if (is_primitive()) return primitive_;
  if (is_null()) return json();
  if (array_) {
    std::vector<json> res;
    for (const auto& item : *array_) {
      res.push_back(item.get<json>());
    }
    return res;
  }
  if (object_) {
    json res = json::object();
    for (const auto& [key, value] : *object_) {
      if (key.is_string()) {
        res[key.get<std::string>()] = value.get<json>();
      } else if (key.is_primitive()) {
        res[key.dump()] = value.get<json>();
      } else {
        throw std::runtime_error("Invalid key type for conversion to JSON: " + key.dump());
      }
    }
    if (is_callable()) {
      res["__callable__"] = true;
    }
    return res;
  }
  throw std::runtime_error("get<json> not defined for this value type: " + dump());
}

} // namespace minja

namespace std {
  template <>
  struct hash<minja::Value> {
    size_t operator()(const minja::Value & v) const {
      if (!v.is_hashable())
        throw std::runtime_error("Unsupported type for hashing: " + v.dump());
      return std::hash<json>()(v.get<json>());
    }
  };
} // namespace std

namespace minja {

static std::string error_location_suffix(const std::string & source, size_t pos) {
  auto get_line = [&](size_t line) {
    auto start = source.begin();
    for (size_t i = 1; i < line; ++i) {
      start = std::find(start, source.end(), '\n') + 1;
    }
    auto end = std::find(start, source.end(), '\n');
    return std::string(start, end);
  };
  auto start = source.begin();
  auto end = source.end();
  auto it = start + pos;
  auto line = std::count(start, it, '\n') + 1;
  auto max_line = std::count(start, end, '\n') + 1;
  auto col = pos - std::string(start, it).rfind('\n');
  std::ostringstream out;
  out << " at row " << line << ", column " << col << ":\n";
  if (line > 1) out << get_line(line - 1) << "\n";
  out << get_line(line) << "\n";
  out << std::string(col - 1, ' ') << "^\n";
  if (line < max_line) out << get_line(line + 1) << "\n";

  return out.str();
}

class Context {
  protected:
    Value values_;
    std::shared_ptr<Context> parent_;
  public:
    Context(Value && values, const std::shared_ptr<Context> & parent = nullptr) : values_(std::move(values)), parent_(parent) {
        if (!values_.is_object()) throw std::runtime_error("Context values must be an object: " + values_.dump());
    }
    virtual ~Context() {}

    static std::shared_ptr<Context> builtins();
    static std::shared_ptr<Context> make(Value && values, const std::shared_ptr<Context> & parent = builtins());

    std::vector<Value> keys() {
        return values_.keys();
    }
    virtual Value get(const Value & key) {
        if (values_.contains(key)) return values_.at(key);
        if (parent_) return parent_->get(key);
        return Value();
    }
    virtual Value & at(const Value & key) {
        if (values_.contains(key)) return values_.at(key);
        if (parent_) return parent_->at(key);
        throw std::runtime_error("Undefined variable: " + key.dump());
    }
    virtual bool contains(const Value & key) {
        if (values_.contains(key)) return true;
        if (parent_) return parent_->contains(key);
        return false;
    }
    virtual void set(const Value & key, const Value & value) {
        values_.set(key, value);
    }
};

struct Location {
    std::shared_ptr<std::string> source;
    size_t pos;
};

class Expression {
protected:
    virtual Value do_evaluate(const std::shared_ptr<Context> & context) const = 0;
public:
    using Parameters = std::vector<std::pair<std::string, std::shared_ptr<Expression>>>;

    Location location;

    Expression(const Location & location) : location(location) {}
    virtual ~Expression() = default;

    Value evaluate(const std::shared_ptr<Context> & context) const {
        try {
            return do_evaluate(context);
        } catch (const std::exception & e) {
            std::ostringstream out;
            out << e.what();
            if (location.source) out << error_location_suffix(*location.source, location.pos);
            throw std::runtime_error(out.str());
        }
    }
};

class VariableExpr : public Expression {
    std::string name;
public:
    VariableExpr(const Location & loc, const std::string& n)
      : Expression(loc), name(n) {}
    std::string get_name() const { return name; }
    Value do_evaluate(const std::shared_ptr<Context> & context) const override {
        if (!context->contains(name)) {
            return Value();
        }
        return context->at(name);
    }
};

static void destructuring_assign(const std::vector<std::string> & var_names, const std::shared_ptr<Context> & context, Value& item) {
  if (var_names.size() == 1) {
      Value name(var_names[0]);
      context->set(name, item);
  } else {
      if (!item.is_array() || item.size() != var_names.size()) {
          throw std::runtime_error("Mismatched number of variables and items in destructuring assignment");
      }
      for (size_t i = 0; i < var_names.size(); ++i) {
          context->set(var_names[i], item.at(i));
      }
  }
}

enum SpaceHandling { Keep, Strip, StripSpaces, StripNewline };

class TemplateToken {
public:
    enum class Type { Text, Expression, If, Else, Elif, EndIf, For, EndFor, Generation, EndGeneration, Set, EndSet, Comment, Macro, EndMacro, Filter, EndFilter, Break, Continue, Call, EndCall };

    static std::string typeToString(Type t) {
        switch (t) {
            case Type::Text: return "text";
            case Type::Expression: return "expression";
            case Type::If: return "if";
            case Type::Else: return "else";
            case Type::Elif: return "elif";
            case Type::EndIf: return "endif";
            case Type::For: return "for";
            case Type::EndFor: return "endfor";
            case Type::Set: return "set";
            case Type::EndSet: return "endset";
            case Type::Comment: return "comment";
            case Type::Macro: return "macro";
            case Type::EndMacro: return "endmacro";
            case Type::Filter: return "filter";
            case Type::EndFilter: return "endfilter";
            case Type::Generation: return "generation";
            case Type::EndGeneration: return "endgeneration";
            case Type::Break: return "break";
            case Type::Continue: return "continue";
            case Type::Call: return "call";
            case Type::EndCall: return "endcall";
        }
        return "Unknown";
    }

    TemplateToken(Type type, const Location & location, SpaceHandling pre, SpaceHandling post) : type(type), location(location), pre_space(pre), post_space(post) {}
    virtual ~TemplateToken() = default;

    Type type;
    Location location;
    SpaceHandling pre_space = SpaceHandling::Keep;
    SpaceHandling post_space = SpaceHandling::Keep;
};

struct TextTemplateToken : public TemplateToken {
    std::string text;
    TextTemplateToken(const Location & loc, SpaceHandling pre, SpaceHandling post, const std::string& t) : TemplateToken(Type::Text, loc, pre, post), text(t) {}
};

struct ExpressionTemplateToken : public TemplateToken {
    std::shared_ptr<Expression> expr;
    ExpressionTemplateToken(const Location & loc, SpaceHandling pre, SpaceHandling post, std::shared_ptr<Expression> && e) : TemplateToken(Type::Expression, loc, pre, post), expr(std::move(e)) {}
};

struct IfTemplateToken : public TemplateToken {
    std::shared_ptr<Expression> condition;
    IfTemplateToken(const Location & loc, SpaceHandling pre, SpaceHandling post, std::shared_ptr<Expression> && c) : TemplateToken(Type::If, loc, pre, post), condition(std::move(c)) {}
};

struct ElifTemplateToken : public TemplateToken {
    std::shared_ptr<Expression> condition;
    ElifTemplateToken(const Location & loc, SpaceHandling pre, SpaceHandling post, std::shared_ptr<Expression> && c) : TemplateToken(Type::Elif, loc, pre, post), condition(std::move(c)) {}
};

struct ElseTemplateToken : public TemplateToken {
    ElseTemplateToken(const Location & loc, SpaceHandling pre, SpaceHandling post) : TemplateToken(Type::Else, loc, pre, post) {}
};

struct EndIfTemplateToken : public TemplateToken {
    EndIfTemplateToken(const Location & loc, SpaceHandling pre, SpaceHandling post) : TemplateToken(Type::EndIf, loc, pre, post) {}
};

struct MacroTemplateToken : public TemplateToken {
    std::shared_ptr<VariableExpr> name;
    Expression::Parameters params;
    MacroTemplateToken(const Location & loc, SpaceHandling pre, SpaceHandling post, std::shared_ptr<VariableExpr> && n, Expression::Parameters && p)
      : TemplateToken(Type::Macro, loc, pre, post), name(std::move(n)), params(std::move(p)) {}
};

struct EndMacroTemplateToken : public TemplateToken {
    EndMacroTemplateToken(const Location & loc, SpaceHandling pre, SpaceHandling post) : TemplateToken(Type::EndMacro, loc, pre, post) {}
};

struct FilterTemplateToken : public TemplateToken {
    std::shared_ptr<Expression> filter;
    FilterTemplateToken(const Location & loc, SpaceHandling pre, SpaceHandling post, std::shared_ptr<Expression> && filter)
      : TemplateToken(Type::Filter, loc, pre, post), filter(std::move(filter)) {}
};

struct EndFilterTemplateToken : public TemplateToken {
    EndFilterTemplateToken(const Location & loc, SpaceHandling pre, SpaceHandling post) : TemplateToken(Type::EndFilter, loc, pre, post) {}
};

struct ForTemplateToken : public TemplateToken {
    std::vector<std::string> var_names;
    std::shared_ptr<Expression> iterable;
    std::shared_ptr<Expression> condition;
    bool recursive;
    ForTemplateToken(const Location & loc, SpaceHandling pre, SpaceHandling post, const std::vector<std::string> & vns, std::shared_ptr<Expression> && iter,
      std::shared_ptr<Expression> && c, bool r)
      : TemplateToken(Type::For, loc, pre, post), var_names(vns), iterable(std::move(iter)), condition(std::move(c)), recursive(r) {}
};

struct EndForTemplateToken : public TemplateToken {
    EndForTemplateToken(const Location & loc, SpaceHandling pre, SpaceHandling post) : TemplateToken(Type::EndFor, loc, pre, post) {}
};

struct GenerationTemplateToken : public TemplateToken {
    GenerationTemplateToken(const Location & loc, SpaceHandling pre, SpaceHandling post) : TemplateToken(Type::Generation, loc, pre, post) {}
};

struct EndGenerationTemplateToken : public TemplateToken {
    EndGenerationTemplateToken(const Location & loc, SpaceHandling pre, SpaceHandling post) : TemplateToken(Type::EndGeneration, loc, pre, post) {}
};

struct SetTemplateToken : public TemplateToken {
    std::string ns;
    std::vector<std::string> var_names;
    std::shared_ptr<Expression> value;
    SetTemplateToken(const Location & loc, SpaceHandling pre, SpaceHandling post, const std::string & ns, const std::vector<std::string> & vns, std::shared_ptr<Expression> && v)
      : TemplateToken(Type::Set, loc, pre, post), ns(ns), var_names(vns), value(std::move(v)) {}
};

struct EndSetTemplateToken : public TemplateToken {
    EndSetTemplateToken(const Location & loc, SpaceHandling pre, SpaceHandling post) : TemplateToken(Type::EndSet, loc, pre, post) {}
};

struct CommentTemplateToken : public TemplateToken {
    std::string text;
    CommentTemplateToken(const Location & loc, SpaceHandling pre, SpaceHandling post, const std::string& t) : TemplateToken(Type::Comment, loc, pre, post), text(t) {}
};

enum class LoopControlType { Break, Continue };

class LoopControlException : public std::runtime_error {
public:
    LoopControlType control_type;
    LoopControlException(const std::string & message, LoopControlType control_type) : std::runtime_error(message), control_type(control_type) {}
    LoopControlException(LoopControlType control_type)
      : std::runtime_error((control_type == LoopControlType::Continue ? "continue" : "break") + std::string(" outside of a loop")),
        control_type(control_type) {}
};

struct LoopControlTemplateToken : public TemplateToken {
    LoopControlType control_type;
    LoopControlTemplateToken(const Location & loc, SpaceHandling pre, SpaceHandling post, LoopControlType control_type) : TemplateToken(Type::Break, loc, pre, post), control_type(control_type) {}
};

struct CallTemplateToken : public TemplateToken {
    std::shared_ptr<Expression> expr;
    CallTemplateToken(const Location & loc, SpaceHandling pre, SpaceHandling post, std::shared_ptr<Expression> && e)
        : TemplateToken(Type::Call, loc, pre, post), expr(std::move(e)) {}
};

struct EndCallTemplateToken : public TemplateToken {
    EndCallTemplateToken(const Location & loc, SpaceHandling pre, SpaceHandling post)
        : TemplateToken(Type::EndCall, loc, pre, post) {}
};

class TemplateNode {
    Location location_;
protected:
    virtual void do_render(std::ostringstream & out, const std::shared_ptr<Context> & context) const = 0;

public:
    TemplateNode(const Location & location) : location_(location) {}
    void render(std::ostringstream & out, const std::shared_ptr<Context> & context) const {
        try {
            do_render(out, context);
        } catch (const LoopControlException & e) {
            // TODO: make stack creation lazy. Only needed if it was thrown outside of a loop.
            std::ostringstream err;
            err << e.what();
            if (location_.source) err << error_location_suffix(*location_.source, location_.pos);
            throw LoopControlException(err.str(), e.control_type);
        } catch (const std::exception & e) {
            std::ostringstream err;
            err << e.what();
            if (location_.source) err << error_location_suffix(*location_.source, location_.pos);
            throw std::runtime_error(err.str());
        }
    }
    const Location & location() const { return location_; }
    virtual ~TemplateNode() = default;
    std::string render(const std::shared_ptr<Context> & context) const {
        std::ostringstream out;
        render(out, context);
        return out.str();
    }
};

class SequenceNode : public TemplateNode {
    std::vector<std::shared_ptr<TemplateNode>> children;
public:
    SequenceNode(const Location & loc, std::vector<std::shared_ptr<TemplateNode>> && c)
      : TemplateNode(loc), children(std::move(c)) {}
    void do_render(std::ostringstream & out, const std::shared_ptr<Context> & context) const override {
        for (const auto& child : children) child->render(out, context);
    }
};

class TextNode : public TemplateNode {
    std::string text;
public:
    TextNode(const Location & loc, const std::string& t) : TemplateNode(loc), text(t) {}
    void do_render(std::ostringstream & out, const std::shared_ptr<Context> &) const override {
      out << text;
    }
};

class ExpressionNode : public TemplateNode {
    std::shared_ptr<Expression> expr;
public:
    ExpressionNode(const Location & loc, std::shared_ptr<Expression> && e) : TemplateNode(loc), expr(std::move(e)) {}
    void do_render(std::ostringstream & out, const std::shared_ptr<Context> & context) const override {
      if (!expr) throw std::runtime_error("ExpressionNode.expr is null");
      auto result = expr->evaluate(context);
      if (result.is_string()) {
          out << result.get<std::string>();
      } else if (result.is_boolean()) {
          out << (result.get<bool>() ? "True" : "False");
      } else if (!result.is_null()) {
          out << result.dump();
      }
  }
};

class IfNode : public TemplateNode {
    std::vector<std::pair<std::shared_ptr<Expression>, std::shared_ptr<TemplateNode>>> cascade;
public:
    IfNode(const Location & loc, std::vector<std::pair<std::shared_ptr<Expression>, std::shared_ptr<TemplateNode>>> && c)
        : TemplateNode(loc), cascade(std::move(c)) {}
    void do_render(std::ostringstream & out, const std::shared_ptr<Context> & context) const override {
      for (const auto& branch : cascade) {
          auto enter_branch = true;
          if (branch.first) {
            enter_branch = branch.first->evaluate(context).to_bool();
          }
          if (enter_branch) {
            if (!branch.second) throw std::runtime_error("IfNode.cascade.second is null");
              branch.second->render(out, context);
              return;
          }
      }
    }
};

class LoopControlNode : public TemplateNode {
    LoopControlType control_type_;
  public:
    LoopControlNode(const Location & loc, LoopControlType control_type) : TemplateNode(loc), control_type_(control_type) {}
    void do_render(std::ostringstream &, const std::shared_ptr<Context> &) const override {
      throw LoopControlException(control_type_);
    }
};

class ForNode : public TemplateNode {
    std::vector<std::string> var_names;
    std::shared_ptr<Expression> iterable;
    std::shared_ptr<Expression> condition;
    std::shared_ptr<TemplateNode> body;
    bool recursive;
    std::shared_ptr<TemplateNode> else_body;
public:
    ForNode(const Location & loc, std::vector<std::string> && var_names, std::shared_ptr<Expression> && iterable,
      std::shared_ptr<Expression> && condition, std::shared_ptr<TemplateNode> && body, bool recursive, std::shared_ptr<TemplateNode> && else_body)
            : TemplateNode(loc), var_names(var_names), iterable(std::move(iterable)), condition(std::move(condition)), body(std::move(body)), recursive(recursive), else_body(std::move(else_body)) {}

    void do_render(std::ostringstream & out, const std::shared_ptr<Context> & context) const override {
      // https://jinja.palletsprojects.com/en/3.0.x/templates/#for
      if (!iterable) throw std::runtime_error("ForNode.iterable is null");
      if (!body) throw std::runtime_error("ForNode.body is null");

      auto iterable_value = iterable->evaluate(context);
      Value::CallableType loop_function;

      std::function<void(Value&)> visit = [&](Value& iter) {
          auto filtered_items = Value::array();
          if (!iter.is_null()) {
            if (!iterable_value.is_iterable()) {
              throw std::runtime_error("For loop iterable must be iterable: " + iterable_value.dump());
            }
            iterable_value.for_each([&](Value & item) {
                destructuring_assign(var_names, context, item);
                if (!condition || condition->evaluate(context).to_bool()) {
                  filtered_items.push_back(item);
                }
            });
          }
          if (filtered_items.empty()) {
            if (else_body) {
              else_body->render(out, context);
            }
          } else {
              auto loop = recursive ? Value::callable(loop_function) : Value::object();
              loop.set("length", (int64_t) filtered_items.size());

              size_t cycle_index = 0;
              loop.set("cycle", Value::callable([&](const std::shared_ptr<Context> &, ArgumentsValue & args) {
                  if (args.args.empty() || !args.kwargs.empty()) {
                      throw std::runtime_error("cycle() expects at least 1 positional argument and no named arg");
                  }
                  auto item = args.args[cycle_index];
                  cycle_index = (cycle_index + 1) % args.args.size();
                  return item;
              }));
              auto loop_context = Context::make(Value::object(), context);
              loop_context->set("loop", loop);
              for (size_t i = 0, n = filtered_items.size(); i < n; ++i) {
                  auto & item = filtered_items.at(i);
                  destructuring_assign(var_names, loop_context, item);
                  loop.set("index", (int64_t) i + 1);
                  loop.set("index0", (int64_t) i);
                  loop.set("revindex", (int64_t) (n - i));
                  loop.set("revindex0", (int64_t) (n - i - 1));
                  loop.set("length", (int64_t) n);
                  loop.set("first", i == 0);
                  loop.set("last", i == (n - 1));
                  loop.set("previtem", i > 0 ? filtered_items.at(i - 1) : Value());
                  loop.set("nextitem", i < n - 1 ? filtered_items.at(i + 1) : Value());
                  try {
                      body->render(out, loop_context);
                  } catch (const LoopControlException & e) {
                      if (e.control_type == LoopControlType::Break) break;
                      if (e.control_type == LoopControlType::Continue) continue;
                  }
              }
          }
      };

      if (recursive) {
        loop_function = [&](const std::shared_ptr<Context> &, ArgumentsValue & args) {
            if (args.args.size() != 1 || !args.kwargs.empty() || !args.args[0].is_array()) {
                throw std::runtime_error("loop() expects exactly 1 positional iterable argument");
            }
            auto & items = args.args[0];
            visit(items);
            return Value();
        };
      }

      visit(iterable_value);
  }
};

class MacroNode : public TemplateNode {
    std::shared_ptr<VariableExpr> name;
    Expression::Parameters params;
    std::shared_ptr<TemplateNode> body;
    std::unordered_map<std::string, size_t> named_param_positions;
public:
    MacroNode(const Location & loc, std::shared_ptr<VariableExpr> && n, Expression::Parameters && p, std::shared_ptr<TemplateNode> && b)
        : TemplateNode(loc), name(std::move(n)), params(std::move(p)), body(std::move(b)) {
        for (size_t i = 0; i < params.size(); ++i) {
          const auto & name = params[i].first;
          if (!name.empty()) {
            named_param_positions[name] = i;
          }
        }
    }
    void do_render(std::ostringstream &, const std::shared_ptr<Context> & context) const override {
        if (!name) throw std::runtime_error("MacroNode.name is null");
        if (!body) throw std::runtime_error("MacroNode.body is null");

        // Use init-capture to avoid dangling 'this' pointer and circular references
        auto callable = Value::callable([weak_context = std::weak_ptr<Context>(context),
                                         name = name, params = params, body = body,
                                         named_param_positions = named_param_positions]
                                        (const std::shared_ptr<Context> & call_context, ArgumentsValue & args) {
            auto context_locked = weak_context.lock();
            if (!context_locked) throw std::runtime_error("Macro context no longer valid");
            auto execution_context = Context::make(Value::object(), context_locked);

            if (call_context->contains("caller")) {
                execution_context->set("caller", call_context->get("caller"));
            }

            std::vector<bool> param_set(params.size(), false);
            for (size_t i = 0, n = args.args.size(); i < n; i++) {
                auto & arg = args.args[i];
                if (i >= params.size()) throw std::runtime_error("Too many positional arguments for macro " + name->get_name());
                param_set[i] = true;
                const auto & param_name = params[i].first;
                execution_context->set(param_name, arg);
            }
            for (auto & [arg_name, value] : args.kwargs) {
                auto it = named_param_positions.find(arg_name);
                if (it == named_param_positions.end()) throw std::runtime_error("Unknown parameter name for macro " + name->get_name() + ": " + arg_name);

                execution_context->set(arg_name, value);
                param_set[it->second] = true;
            }
            // Set default values for parameters that were not passed
            for (size_t i = 0, n = params.size(); i < n; i++) {
                if (!param_set[i] && params[i].second != nullptr) {
                    auto val = params[i].second->evaluate(call_context);
                    execution_context->set(params[i].first, val);
                }
            }
            return body->render(execution_context);
        });
        context->set(name->get_name(), callable);
    }
};

class FilterNode : public TemplateNode {
    std::shared_ptr<Expression> filter;
    std::shared_ptr<TemplateNode> body;

public:
    FilterNode(const Location & loc, std::shared_ptr<Expression> && f, std::shared_ptr<TemplateNode> && b)
        : TemplateNode(loc), filter(std::move(f)), body(std::move(b)) {}

    void do_render(std::ostringstream & out, const std::shared_ptr<Context> & context) const override {
        if (!filter) throw std::runtime_error("FilterNode.filter is null");
        if (!body) throw std::runtime_error("FilterNode.body is null");
        auto filter_value = filter->evaluate(context);
        if (!filter_value.is_callable()) {
            throw std::runtime_error("Filter must be a callable: " + filter_value.dump());
        }
        std::string rendered_body = body->render(context);

        ArgumentsValue filter_args = {{Value(rendered_body)}, {}};
        auto result = filter_value.call(context, filter_args);
        out << result.to_str();
    }
};

class SetNode : public TemplateNode {
    std::string ns;
    std::vector<std::string> var_names;
    std::shared_ptr<Expression> value;
public:
    SetNode(const Location & loc, const std::string & ns, const std::vector<std::string> & vns, std::shared_ptr<Expression> && v)
        : TemplateNode(loc), ns(ns), var_names(vns), value(std::move(v)) {}
    void do_render(std::ostringstream &, const std::shared_ptr<Context> & context) const override {
      if (!value) throw std::runtime_error("SetNode.value is null");
      if (!ns.empty()) {
        if (var_names.size() != 1) {
          throw std::runtime_error("Namespaced set only supports a single variable name");
        }
        auto & name = var_names[0];
        auto ns_value = context->get(ns);
        if (!ns_value.is_object()) throw std::runtime_error("Namespace '" + ns + "' is not an object");
        ns_value.set(name, this->value->evaluate(context));
      } else {
        auto val = value->evaluate(context);
        destructuring_assign(var_names, context, val);
      }
    }
};

class SetTemplateNode : public TemplateNode {
    std::string name;
    std::shared_ptr<TemplateNode> template_value;
public:
    SetTemplateNode(const Location & loc, const std::string & name, std::shared_ptr<TemplateNode> && tv)
        : TemplateNode(loc), name(name), template_value(std::move(tv)) {}
    void do_render(std::ostringstream &, const std::shared_ptr<Context> & context) const override {
      if (!template_value) throw std::runtime_error("SetTemplateNode.template_value is null");
      Value value { template_value->render(context) };
      context->set(name, value);
    }
};

class IfExpr : public Expression {
    std::shared_ptr<Expression> condition;
    std::shared_ptr<Expression> then_expr;
    std::shared_ptr<Expression> else_expr;
public:
    IfExpr(const Location & loc, std::shared_ptr<Expression> && c, std::shared_ptr<Expression> && t, std::shared_ptr<Expression> && e)
        : Expression(loc), condition(std::move(c)), then_expr(std::move(t)), else_expr(std::move(e)) {}
    Value do_evaluate(const std::shared_ptr<Context> & context) const override {
      if (!condition) throw std::runtime_error("IfExpr.condition is null");
      if (!then_expr) throw std::runtime_error("IfExpr.then_expr is null");
      if (condition->evaluate(context).to_bool()) {
        return then_expr->evaluate(context);
      }
      if (else_expr) {
        return else_expr->evaluate(context);
      }
      return nullptr;
    }
};

class LiteralExpr : public Expression {
    Value value;
public:
    LiteralExpr(const Location & loc, const Value& v)
      : Expression(loc), value(v) {}
    Value do_evaluate(const std::shared_ptr<Context> &) const override { return value; }
};

class ArrayExpr : public Expression {
    std::vector<std::shared_ptr<Expression>> elements;
public:
    ArrayExpr(const Location & loc, std::vector<std::shared_ptr<Expression>> && e)
      : Expression(loc), elements(std::move(e)) {}
    Value do_evaluate(const std::shared_ptr<Context> & context) const override {
        auto result = Value::array();
        for (const auto& e : elements) {
            if (!e) throw std::runtime_error("Array element is null");
            result.push_back(e->evaluate(context));
        }
        return result;
    }
};

class DictExpr : public Expression {
    std::vector<std::pair<std::shared_ptr<Expression>, std::shared_ptr<Expression>>> elements;
public:
    DictExpr(const Location & loc, std::vector<std::pair<std::shared_ptr<Expression>, std::shared_ptr<Expression>>> && e)
      : Expression(loc), elements(std::move(e)) {}
    Value do_evaluate(const std::shared_ptr<Context> & context) const override {
        auto result = Value::object();
        for (const auto& [key, value] : elements) {
            if (!key) throw std::runtime_error("Dict key is null");
            if (!value) throw std::runtime_error("Dict value is null");
            result.set(key->evaluate(context), value->evaluate(context));
        }
        return result;
    }
};

class SliceExpr : public Expression {
public:
    std::shared_ptr<Expression> start, end, step;
    SliceExpr(const Location & loc, std::shared_ptr<Expression> && s, std::shared_ptr<Expression> && e, std::shared_ptr<Expression> && st = nullptr)
      : Expression(loc), start(std::move(s)), end(std::move(e)), step(std::move(st)) {}
    Value do_evaluate(const std::shared_ptr<Context> &) const override {
        throw std::runtime_error("SliceExpr not implemented");
    }
};

class SubscriptExpr : public Expression {
    std::shared_ptr<Expression> base;
    std::shared_ptr<Expression> index;
public:
    SubscriptExpr(const Location & loc, std::shared_ptr<Expression> && b, std::shared_ptr<Expression> && i)
        : Expression(loc), base(std::move(b)), index(std::move(i)) {}
    Value do_evaluate(const std::shared_ptr<Context> & context) const override {
        if (!base) throw std::runtime_error("SubscriptExpr.base is null");
        if (!index) throw std::runtime_error("SubscriptExpr.index is null");
        auto target_value = base->evaluate(context);
        if (auto slice = dynamic_cast<SliceExpr*>(index.get())) {
          auto len = target_value.size();
          auto wrap = [len](int64_t i) -> int64_t {
            if (i < 0) {
              return i + len;
            }
            return i;
          };
          int64_t step = slice->step ? slice->step->evaluate(context).get<int64_t>() : 1;
          if (!step) {
            throw std::runtime_error("slice step cannot be zero");
          }
          int64_t start = slice->start ? wrap(slice->start->evaluate(context).get<int64_t>()) : (step < 0 ? len - 1 : 0);
          int64_t end = slice->end ? wrap(slice->end->evaluate(context).get<int64_t>()) : (step < 0 ? -1 : len);
          if (target_value.is_string()) {
            std::string s = target_value.get<std::string>();

            std::string result;
            if (start < end && step == 1) {
              result = s.substr(start, end - start);
            } else {
              for (int64_t i = start; step > 0 ? i < end : i > end; i += step) {
                result += s[i];
              }
            }
            return result;

          } else if (target_value.is_array()) {
            auto result = Value::array();
            for (int64_t i = start; step > 0 ? i < end : i > end; i += step) {
              result.push_back(target_value.at(i));
            }
            return result;
          } else {
            throw std::runtime_error(target_value.is_null() ? "Cannot subscript null" : "Subscripting only supported on arrays and strings");
          }
        } else {
          auto index_value = index->evaluate(context);
          if (target_value.is_null()) {
            if (auto t = dynamic_cast<VariableExpr*>(base.get())) {
              throw std::runtime_error("'" + t->get_name() + "' is " + (context->contains(t->get_name()) ? "null" : "not defined"));
            }
            throw std::runtime_error("Trying to access property '" +  index_value.dump() + "' on null!");
          }
          return target_value.get(index_value);
        }
    }
};

class UnaryOpExpr : public Expression {
public:
    enum class Op { Plus, Minus, LogicalNot, Expansion, ExpansionDict };
    std::shared_ptr<Expression> expr;
    Op op;
    UnaryOpExpr(const Location & loc, std::shared_ptr<Expression> && e, Op o)
      : Expression(loc), expr(std::move(e)), op(o) {}
    Value do_evaluate(const std::shared_ptr<Context> & context) const override {
        if (!expr) throw std::runtime_error("UnaryOpExpr.expr is null");
        auto e = expr->evaluate(context);
        switch (op) {
            case Op::Plus: return e;
            case Op::Minus: return -e;
            case Op::LogicalNot: return !e.to_bool();
            case Op::Expansion:
            case Op::ExpansionDict:
                throw std::runtime_error("Expansion operator is only supported in function calls and collections");

        }
        throw std::runtime_error("Unknown unary operator");
    }
};

static bool in(const Value & value, const Value & container) {
  return (((container.is_array() || container.is_object()) && container.contains(value)) ||
      (value.is_string() && container.is_string() &&
        container.to_str().find(value.to_str()) != std::string::npos));
}

class BinaryOpExpr : public Expression {
public:
    enum class Op { StrConcat, Add, Sub, Mul, MulMul, Div, DivDiv, Mod, Eq, Ne, Lt, Gt, Le, Ge, And, Or, In, NotIn, Is, IsNot };
private:
    std::shared_ptr<Expression> left;
    std::shared_ptr<Expression> right;
    Op op;
public:
    BinaryOpExpr(const Location & loc, std::shared_ptr<Expression> && l, std::shared_ptr<Expression> && r, Op o)
        : Expression(loc), left(std::move(l)), right(std::move(r)), op(o) {}
    Value do_evaluate(const std::shared_ptr<Context> & context) const override {
        if (!left) throw std::runtime_error("BinaryOpExpr.left is null");
        if (!right) throw std::runtime_error("BinaryOpExpr.right is null");
        auto l = left->evaluate(context);

        auto do_eval = [&](const Value & l) -> Value {
          if (op == Op::Is || op == Op::IsNot) {
            auto t = dynamic_cast<VariableExpr*>(right.get());
            if (!t) throw std::runtime_error("Right side of 'is' operator must be a variable");

            auto eval = [&]() {
              const auto & name = t->get_name();
              if (name == "none") return l.is_null();
              if (name == "boolean") return l.is_boolean();
              if (name == "integer") return l.is_number_integer();
              if (name == "float") return l.is_number_float();
              if (name == "number") return l.is_number();
              if (name == "string") return l.is_string();
              if (name == "mapping") return l.is_object();
              if (name == "iterable") return l.is_iterable();
              if (name == "sequence") return l.is_array();
              if (name == "defined") return !l.is_null();
              if (name == "true") return l.to_bool();
              if (name == "false") return !l.to_bool();
              throw std::runtime_error("Unknown type for 'is' operator: " + name);
            };
            auto value = eval();
            return Value(op == Op::Is ? value : !value);
          }

          if (op == Op::And) {
            if (!l.to_bool()) return Value(false);
            return right->evaluate(context).to_bool();
          } else if (op == Op::Or) {
            if (l.to_bool()) return l;
            return right->evaluate(context);
          }

          auto r = right->evaluate(context);
          switch (op) {
              case Op::StrConcat: return l.to_str() + r.to_str();
              case Op::Add:       return l + r;
              case Op::Sub:       return l - r;
              case Op::Mul:       return l * r;
              case Op::Div:       return l / r;
              case Op::MulMul:    return std::pow(l.get<double>(), r.get<double>());
              case Op::DivDiv:    return l.get<int64_t>() / r.get<int64_t>();
              case Op::Mod:       return l.get<int64_t>() % r.get<int64_t>();
              case Op::Eq:        return l == r;
              case Op::Ne:        return l != r;
              case Op::Lt:        return l < r;
              case Op::Gt:        return l > r;
              case Op::Le:        return l <= r;
              case Op::Ge:        return l >= r;
              case Op::In:        return in(l, r);
              case Op::NotIn:     return !in(l, r);
              default:            break;
          }
          throw std::runtime_error("Unknown binary operator");
        };

        if (l.is_callable()) {
          return Value::callable([l, do_eval](const std::shared_ptr<Context> & context, ArgumentsValue & args) {
            auto ll = l.call(context, args);
            return do_eval(ll); //args[0].second);
          });
        } else {
          return do_eval(l);
        }
    }
};

struct ArgumentsExpression {
    std::vector<std::shared_ptr<Expression>> args;
    std::vector<std::pair<std::string, std::shared_ptr<Expression>>> kwargs;

    ArgumentsValue evaluate(const std::shared_ptr<Context> & context) const {
        ArgumentsValue vargs;
        for (const auto& arg : this->args) {
            if (auto un_expr = std::dynamic_pointer_cast<UnaryOpExpr>(arg)) {
                if (un_expr->op == UnaryOpExpr::Op::Expansion) {
                    auto array = un_expr->expr->evaluate(context);
                    if (!array.is_array()) {
                        throw std::runtime_error("Expansion operator only supported on arrays");
                    }
                    array.for_each([&](Value & value) {
                        vargs.args.push_back(value);
                    });
                    continue;
                } else if (un_expr->op == UnaryOpExpr::Op::ExpansionDict) {
                    auto dict = un_expr->expr->evaluate(context);
                    if (!dict.is_object()) {
                        throw std::runtime_error("ExpansionDict operator only supported on objects");
                    }
                    dict.for_each([&](const Value & key) {
                        vargs.kwargs.push_back({key.get<std::string>(), dict.at(key)});
                    });
                    continue;
                }
            }
            vargs.args.push_back(arg->evaluate(context));
        }
        for (const auto& [name, value] : this->kwargs) {
            vargs.kwargs.push_back({name, value->evaluate(context)});
        }
        return vargs;
    }
};

static std::string strip(const std::string & s, const std::string & chars = "", bool left = true, bool right = true) {
  auto charset = chars.empty() ? " \t\n\r" : chars;
  auto start = left ? s.find_first_not_of(charset) : 0;
  if (start == std::string::npos) return "";
  auto end = right ? s.find_last_not_of(charset) : s.size() - 1;
  return s.substr(start, end - start + 1);
}

static std::vector<std::string> split(const std::string & s, const std::string & sep) {
  std::vector<std::string> result;
  size_t start = 0;
  size_t end = s.find(sep);
  while (end != std::string::npos) {
    result.push_back(s.substr(start, end - start));
    start = end + sep.length();
    end = s.find(sep, start);
  }
  result.push_back(s.substr(start));
  return result;
}

static std::string capitalize(const std::string & s) {
  if (s.empty()) return s;
  auto result = s;
  result[0] = std::toupper(result[0]);
  return result;
}

static std::string html_escape(const std::string & s) {
  std::string result;
  result.reserve(s.size());
  for (const auto & c : s) {
    switch (c) {
      case '&': result += "&amp;"; break;
      case '<': result += "&lt;"; break;
      case '>': result += "&gt;"; break;
      case '"': result += "&#34;"; break;
      case '\'': result += "&apos;"; break;
      default: result += c; break;
    }
  }
  return result;
}

class MethodCallExpr : public Expression {
    std::shared_ptr<Expression> object;
    std::shared_ptr<VariableExpr> method;
    ArgumentsExpression args;
public:
    MethodCallExpr(const Location & loc, std::shared_ptr<Expression> && obj, std::shared_ptr<VariableExpr> && m, ArgumentsExpression && a)
        : Expression(loc), object(std::move(obj)), method(std::move(m)), args(std::move(a)) {}
    Value do_evaluate(const std::shared_ptr<Context> & context) const override {
        if (!object) throw std::runtime_error("MethodCallExpr.object is null");
        if (!method) throw std::runtime_error("MethodCallExpr.method is null");
        auto obj = object->evaluate(context);
        auto vargs = args.evaluate(context);
        if (obj.is_null()) {
          throw std::runtime_error("Trying to call method '" + method->get_name() + "' on null");
        }
        if (obj.is_array()) {
          if (method->get_name() == "append") {
              vargs.expectArgs("append method", {1, 1}, {0, 0});
              obj.push_back(vargs.args[0]);
              return Value();
          } else if (method->get_name() == "pop") {
              vargs.expectArgs("pop method", {0, 1}, {0, 0});
              return obj.pop(vargs.args.empty() ? Value() : vargs.args[0]);
          } else if (method->get_name() == "insert") {
              vargs.expectArgs("insert method", {2, 2}, {0, 0});
              auto index = vargs.args[0].get<int64_t>();
              if (index < 0 || index > (int64_t) obj.size()) throw std::runtime_error("Index out of range for insert method");
              obj.insert(index, vargs.args[1]);
              return Value();
          }
        } else if (obj.is_object()) {
          if (method->get_name() == "items") {
            vargs.expectArgs("items method", {0, 0}, {0, 0});
            auto result = Value::array();
            for (const auto& key : obj.keys()) {
              result.push_back(Value::array({key, obj.at(key)}));
            }
            return result;
          } else if (method->get_name() == "pop") {
            vargs.expectArgs("pop method", {1, 1}, {0, 0});
            return obj.pop(vargs.args[0]);
          } else if (method->get_name() == "keys") {
            vargs.expectArgs("keys method", {0, 0}, {0, 0});
            auto result = Value::array();
            for (const auto& key : obj.keys()) {
              result.push_back(Value(key));
            }
            return result;
          } else if (method->get_name() == "get") {
            vargs.expectArgs("get method", {1, 2}, {0, 0});
            auto key = vargs.args[0];
            if (vargs.args.size() == 1) {
              return obj.contains(key) ? obj.at(key) : Value();
            } else {
              return obj.contains(key) ? obj.at(key) : vargs.args[1];
            }
          } else if (obj.contains(method->get_name())) {
            auto callable = obj.at(method->get_name());
            if (!callable.is_callable()) {
              throw std::runtime_error("Property '" + method->get_name() + "' is not callable");
            }
            return callable.call(context, vargs);
          }
        } else if (obj.is_string()) {
          auto str = obj.get<std::string>();
          if (method->get_name() == "strip") {
            vargs.expectArgs("strip method", {0, 1}, {0, 0});
            auto chars = vargs.args.empty() ? "" : vargs.args[0].get<std::string>();
            return Value(strip(str, chars));
          } else if (method->get_name() == "lstrip") {
            vargs.expectArgs("lstrip method", {0, 1}, {0, 0});
            auto chars = vargs.args.empty() ? "" : vargs.args[0].get<std::string>();
            return Value(strip(str, chars, /* left= */ true, /* right= */ false));
          } else if (method->get_name() == "rstrip") {
            vargs.expectArgs("rstrip method", {0, 1}, {0, 0});
            auto chars = vargs.args.empty() ? "" : vargs.args[0].get<std::string>();
            return Value(strip(str, chars, /* left= */ false, /* right= */ true));
          } else if (method->get_name() == "split") {
            vargs.expectArgs("split method", {1, 1}, {0, 0});
            auto sep = vargs.args[0].get<std::string>();
            auto parts = split(str, sep);
            Value result = Value::array();
            for (const auto& part : parts) {
              result.push_back(Value(part));
            }
            return result;
          } else if (method->get_name() == "capitalize") {
            vargs.expectArgs("capitalize method", {0, 0}, {0, 0});
            return Value(capitalize(str));
          } else if (method->get_name() == "upper") {
            vargs.expectArgs("upper method", {0, 0}, {0, 0});
            auto result = str;
            std::transform(result.begin(), result.end(), result.begin(), ::toupper);
            return Value(result);
          } else if (method->get_name() == "lower") {
            vargs.expectArgs("lower method", {0, 0}, {0, 0});
            auto result = str;
            std::transform(result.begin(), result.end(), result.begin(), ::tolower);
            return Value(result);
          } else if (method->get_name() == "endswith") {
            vargs.expectArgs("endswith method", {1, 1}, {0, 0});
            auto suffix = vargs.args[0].get<std::string>();
            return suffix.length() <= str.length() && std::equal(suffix.rbegin(), suffix.rend(), str.rbegin());
          } else if (method->get_name() == "startswith") {
            vargs.expectArgs("startswith method", {1, 1}, {0, 0});
            auto prefix = vargs.args[0].get<std::string>();
            return prefix.length() <= str.length() && std::equal(prefix.begin(), prefix.end(), str.begin());
          } else if (method->get_name() == "title") {
            vargs.expectArgs("title method", {0, 0}, {0, 0});
            auto res = str;
            for (size_t i = 0, n = res.size(); i < n; ++i) {
              if (i == 0 || std::isspace(res[i - 1])) res[i] = std::toupper(res[i]);
              else res[i] = std::tolower(res[i]);
            }
            return res;
          } else if (method->get_name() == "replace") {
            vargs.expectArgs("replace method", {2, 3}, {0, 0});
            auto before = vargs.args[0].get<std::string>();
            auto after = vargs.args[1].get<std::string>();
            auto count = vargs.args.size() == 3 ? vargs.args[2].get<int64_t>()
                                                : str.length();
            size_t start_pos = 0;
            while ((start_pos = str.find(before, start_pos)) != std::string::npos &&
                  count-- > 0) {
              str.replace(start_pos, before.length(), after);
              start_pos += after.length();
            }
            return str;
          }
        }
        throw std::runtime_error("Unknown method: " + method->get_name());
    }
};

class CallExpr : public Expression {
public:
    std::shared_ptr<Expression> object;
    ArgumentsExpression args;
    CallExpr(const Location & loc, std::shared_ptr<Expression> && obj, ArgumentsExpression && a)
        : Expression(loc), object(std::move(obj)), args(std::move(a)) {}
    Value do_evaluate(const std::shared_ptr<Context> & context) const override {
        if (!object) throw std::runtime_error("CallExpr.object is null");
        auto obj = object->evaluate(context);
        if (!obj.is_callable()) {
          throw std::runtime_error("Object is not callable: " + obj.dump(2));
        }
        auto vargs = args.evaluate(context);
        return obj.call(context, vargs);
    }
};

class CallNode : public TemplateNode {
    std::shared_ptr<Expression> expr;
    std::shared_ptr<TemplateNode> body;

public:
    CallNode(const Location & loc, std::shared_ptr<Expression> && e, std::shared_ptr<TemplateNode> && b)
        : TemplateNode(loc), expr(std::move(e)), body(std::move(b)) {}

    void do_render(std::ostringstream & out, const std::shared_ptr<Context> & context) const override {
        if (!expr) throw std::runtime_error("CallNode.expr is null");
        if (!body) throw std::runtime_error("CallNode.body is null");

        // Use init-capture to avoid dangling 'this' pointer and circular references
        auto caller = Value::callable([weak_context = std::weak_ptr<Context>(context), body=body]
                                      (const std::shared_ptr<Context> &, ArgumentsValue &) -> Value {
            auto context_locked = weak_context.lock();
            if (!context_locked) throw std::runtime_error("Caller context no longer valid");
            return Value(body->render(context_locked));
        });

        context->set("caller", caller);

        auto call_expr = dynamic_cast<CallExpr*>(expr.get());
        if (!call_expr) {
            throw std::runtime_error("Invalid call block syntax - expected function call");
        }

        Value function = call_expr->object->evaluate(context);
        if (!function.is_callable()) {
            throw std::runtime_error("Call target must be callable: " + function.dump());
        }
        ArgumentsValue args = call_expr->args.evaluate(context);

        Value result = function.call(context, args);
        out << result.to_str();
    }
};

class FilterExpr : public Expression {
    std::vector<std::shared_ptr<Expression>> parts;
public:
    FilterExpr(const Location & loc, std::vector<std::shared_ptr<Expression>> && p)
      : Expression(loc), parts(std::move(p)) {}
    Value do_evaluate(const std::shared_ptr<Context> & context) const override {
        Value result;
        bool first = true;
        for (const auto& part : parts) {
          if (!part) throw std::runtime_error("FilterExpr.part is null");
          if (first) {
            first = false;
            result = part->evaluate(context);
          } else {
            if (auto ce = dynamic_cast<CallExpr*>(part.get())) {
              auto target = ce->object->evaluate(context);
              ArgumentsValue args = ce->args.evaluate(context);
              args.args.insert(args.args.begin(), result);
              result = target.call(context, args);
            } else {
              auto callable = part->evaluate(context);
              ArgumentsValue args;
              args.args.insert(args.args.begin(), result);
              result = callable.call(context, args);
            }
          }
        }
        return result;
    }

    void prepend(std::shared_ptr<Expression> && e) {
        parts.insert(parts.begin(), std::move(e));
    }
};

class Parser {
private:
    using CharIterator = std::string::const_iterator;

    std::shared_ptr<std::string> template_str;
    CharIterator start, end, it;
    Options options;

    Parser(const std::shared_ptr<std::string>& template_str, const Options & options) : template_str(template_str), options(options) {
      if (!template_str) throw std::runtime_error("Template string is null");
      start = it = this->template_str->begin();
      end = this->template_str->end();
    }

    bool consumeSpaces(SpaceHandling space_handling = SpaceHandling::Strip) {
      if (space_handling == SpaceHandling::Strip) {
        while (it != end && std::isspace(*it)) ++it;
      }
      return true;
    }

    std::unique_ptr<std::string> parseString() {
      auto doParse = [&](char quote) -> std::unique_ptr<std::string> {
        if (it == end || *it != quote) return nullptr;
        std::string result;
        bool escape = false;
        for (++it; it != end; ++it) {
          if (escape) {
            escape = false;
            switch (*it) {
              case 'n': result += '\n'; break;
              case 'r': result += '\r'; break;
              case 't': result += '\t'; break;
              case 'b': result += '\b'; break;
              case 'f': result += '\f'; break;
              case '\\': result += '\\'; break;
              default:
                if (*it == quote) {
                  result += quote;
                } else {
                  result += *it;
                }
                break;
            }
          } else if (*it == '\\') {
            escape = true;
          } else if (*it == quote) {
              ++it;
            return std::make_unique<std::string>(std::move(result));
          } else {
            result += *it;
          }
        }
        return nullptr;
      };

      consumeSpaces();
      if (it == end) return nullptr;
      if (*it == '"') return doParse('"');
      if (*it == '\'') return doParse('\'');
      return nullptr;
    }

    json parseNumber(CharIterator& it, const CharIterator& end) {
        auto before = it;
        consumeSpaces();
        auto start = it;
        bool hasDecimal = false;
        bool hasExponent = false;

        if (it != end && (*it == '-' || *it == '+')) ++it;

        while (it != end) {
          if (std::isdigit(*it)) {
            ++it;
          } else if (*it == '.') {
            if (hasDecimal) throw std::runtime_error("Multiple decimal points");
            hasDecimal = true;
            ++it;
          } else if (it != start && (*it == 'e' || *it == 'E')) {
            if (hasExponent) throw std::runtime_error("Multiple exponents");
            hasExponent = true;
            ++it;
          } else {
            break;
          }
        }
        if (start == it) {
          it = before;
          return json(); // No valid characters found
        }

        std::string str(start, it);
        try {
          return json::parse(str);
        } catch (json::parse_error& e) {
          throw std::runtime_error("Failed to parse number: '" + str + "' (" + std::string(e.what()) + ")");
          return json();
        }
    }

    /** integer, float, bool, string */
    std::shared_ptr<Value> parseConstant() {
      auto start = it;
      consumeSpaces();
      if (it == end) return nullptr;
      if (*it == '"' || *it == '\'') {
        auto str = parseString();
        if (str) return std::make_shared<Value>(*str);
      }
      static std::regex prim_tok(R"(true\b|True\b|false\b|False\b|None\b)");
      auto token = consumeToken(prim_tok);
      if (!token.empty()) {
        if (token == "true" || token == "True") return std::make_shared<Value>(true);
        if (token == "false" || token == "False") return std::make_shared<Value>(false);
        if (token == "None") return std::make_shared<Value>(nullptr);
        throw std::runtime_error("Unknown constant token: " + token);
      }

      auto number = parseNumber(it, end);
      if (!number.is_null()) return std::make_shared<Value>(number);

      it = start;
      return nullptr;
    }

    class expression_parsing_error : public std::runtime_error {
        const CharIterator it;
      public:
        expression_parsing_error(const std::string & message, const CharIterator & it)
            : std::runtime_error(message), it(it) {}
        size_t get_pos(const CharIterator & begin) const {
            return std::distance(begin, it);
      }
    };

    bool peekSymbols(const std::vector<std::string> & symbols) const {
        for (const auto & symbol : symbols) {
            if (std::distance(it, end) >= (int64_t) symbol.size() && std::string(it, it + symbol.size()) == symbol) {
                return true;
            }
        }
        return false;
    }

    std::vector<std::string> consumeTokenGroups(const std::regex & regex, SpaceHandling space_handling = SpaceHandling::Strip) {
        auto start = it;
        consumeSpaces(space_handling);
        std::smatch match;
        if (std::regex_search(it, end, match, regex) && match.position() == 0) {
            it += match[0].length();
            std::vector<std::string> ret;
            for (size_t i = 0, n = match.size(); i < n; ++i) {
                ret.push_back(match[i].str());
            }
            return ret;
        }
        it = start;
        return {};
    }
    std::string consumeToken(const std::regex & regex, SpaceHandling space_handling = SpaceHandling::Strip) {
        auto start = it;
        consumeSpaces(space_handling);
        std::smatch match;
        if (std::regex_search(it, end, match, regex) && match.position() == 0) {
            it += match[0].length();
            return match[0].str();
        }
        it = start;
        return "";
    }

    std::string consumeToken(const std::string & token, SpaceHandling space_handling = SpaceHandling::Strip) {
        auto start = it;
        consumeSpaces(space_handling);
        if (std::distance(it, end) >= (int64_t) token.size() && std::string(it, it + token.size()) == token) {
            it += token.size();
            return token;
        }
        it = start;
        return "";
    }

    std::shared_ptr<Expression> parseExpression(bool allow_if_expr = true) {
        auto left = parseLogicalOr();
        if (it == end) return left;

        if (!allow_if_expr) return left;

        static std::regex if_tok(R"(if\b)");
        if (consumeToken(if_tok).empty()) {
          return left;
        }

        auto location = get_location();
        auto [condition, else_expr] = parseIfExpression();
        return std::make_shared<IfExpr>(location, std::move(condition), std::move(left), std::move(else_expr));
    }

    Location get_location() const {
        return {template_str, (size_t) std::distance(start, it)};
    }

    std::pair<std::shared_ptr<Expression>, std::shared_ptr<Expression>> parseIfExpression() {
        auto condition = parseLogicalOr();
        if (!condition) throw std::runtime_error("Expected condition expression");

        static std::regex else_tok(R"(else\b)");
        std::shared_ptr<Expression> else_expr;
        if (!consumeToken(else_tok).empty()) {
          else_expr = parseExpression();
          if (!else_expr) throw std::runtime_error("Expected 'else' expression");
        }
        return std::pair(std::move(condition), std::move(else_expr));
    }

    std::shared_ptr<Expression> parseLogicalOr() {
        auto left = parseLogicalAnd();
        if (!left) throw std::runtime_error("Expected left side of 'logical or' expression");

        static std::regex or_tok(R"(or\b)");
        auto location = get_location();
        while (!consumeToken(or_tok).empty()) {
            auto right = parseLogicalAnd();
            if (!right) throw std::runtime_error("Expected right side of 'or' expression");
            left = std::make_shared<BinaryOpExpr>(location, std::move(left), std::move(right), BinaryOpExpr::Op::Or);
        }
        return left;
    }

    std::shared_ptr<Expression> parseLogicalNot() {
        static std::regex not_tok(R"(not\b)");
        auto location = get_location();

        if (!consumeToken(not_tok).empty()) {
          auto sub = parseLogicalNot();
          if (!sub) throw std::runtime_error("Expected expression after 'not' keyword");
          return std::make_shared<UnaryOpExpr>(location, std::move(sub), UnaryOpExpr::Op::LogicalNot);
        }
        return parseLogicalCompare();
    }

    std::shared_ptr<Expression> parseLogicalAnd() {
        auto left = parseLogicalNot();
        if (!left) throw std::runtime_error("Expected left side of 'logical and' expression");

        static std::regex and_tok(R"(and\b)");
        auto location = get_location();
        while (!consumeToken(and_tok).empty()) {
            auto right = parseLogicalNot();
            if (!right) throw std::runtime_error("Expected right side of 'and' expression");
            left = std::make_shared<BinaryOpExpr>(location, std::move(left), std::move(right), BinaryOpExpr::Op::And);
        }
        return left;
    }

    std::shared_ptr<Expression> parseLogicalCompare() {
        auto left = parseStringConcat();
        if (!left) throw std::runtime_error("Expected left side of 'logical compare' expression");

        static std::regex compare_tok(R"(==|!=|<=?|>=?|in\b|is\b|not\s+in\b)");
        static std::regex not_tok(R"(not\b)");
        std::string op_str;
        while (!(op_str = consumeToken(compare_tok)).empty()) {
            auto location = get_location();
            if (op_str == "is") {
              auto negated = !consumeToken(not_tok).empty();

              auto identifier = parseIdentifier();
              if (!identifier) throw std::runtime_error("Expected identifier after 'is' keyword");

              return std::make_shared<BinaryOpExpr>(
                  left->location,
                  std::move(left), std::move(identifier),
                  negated ? BinaryOpExpr::Op::IsNot : BinaryOpExpr::Op::Is);
            }
            auto right = parseStringConcat();
            if (!right) throw std::runtime_error("Expected right side of 'logical compare' expression");
            BinaryOpExpr::Op op;
            if (op_str == "==") op = BinaryOpExpr::Op::Eq;
            else if (op_str == "!=") op = BinaryOpExpr::Op::Ne;
            else if (op_str == "<") op = BinaryOpExpr::Op::Lt;
            else if (op_str == ">") op = BinaryOpExpr::Op::Gt;
            else if (op_str == "<=") op = BinaryOpExpr::Op::Le;
            else if (op_str == ">=") op = BinaryOpExpr::Op::Ge;
            else if (op_str == "in") op = BinaryOpExpr::Op::In;
            else if (op_str.substr(0, 3) == "not") op = BinaryOpExpr::Op::NotIn;
            else throw std::runtime_error("Unknown comparison operator: " + op_str);
            left = std::make_shared<BinaryOpExpr>(get_location(), std::move(left), std::move(right), op);
        }
        return left;
    }

    Expression::Parameters parseParameters() {
        consumeSpaces();
        if (consumeToken("(").empty()) throw std::runtime_error("Expected opening parenthesis in param list");

        Expression::Parameters result;

        while (it != end) {
            if (!consumeToken(")").empty()) {
                return result;
            }
            auto expr = parseExpression();
            if (!expr) throw std::runtime_error("Expected expression in call args");

            if (auto ident = dynamic_cast<VariableExpr*>(expr.get())) {
                if (!consumeToken("=").empty()) {
                    auto value = parseExpression();
                    if (!value) throw std::runtime_error("Expected expression in for named arg");
                    result.emplace_back(ident->get_name(), std::move(value));
                } else {
                    result.emplace_back(ident->get_name(), nullptr);
                }
            } else {
                result.emplace_back(std::string(), std::move(expr));
            }
            if (consumeToken(",").empty()) {
              if (consumeToken(")").empty()) {
                throw std::runtime_error("Expected closing parenthesis in call args");
              }
              return result;
            }
        }
        throw std::runtime_error("Expected closing parenthesis in call args");
    }

    ArgumentsExpression parseCallArgs() {
        consumeSpaces();
        if (consumeToken("(").empty()) throw std::runtime_error("Expected opening parenthesis in call args");

        ArgumentsExpression result;

        while (it != end) {
            if (!consumeToken(")").empty()) {
                return result;
            }
            auto expr = parseExpression();
            if (!expr) throw std::runtime_error("Expected expression in call args");

            if (auto ident = dynamic_cast<VariableExpr*>(expr.get())) {
                if (!consumeToken("=").empty()) {
                    auto value = parseExpression();
                    if (!value) throw std::runtime_error("Expected expression in for named arg");
                    result.kwargs.emplace_back(ident->get_name(), std::move(value));
                } else {
                    result.args.emplace_back(std::move(expr));
                }
            } else {
                result.args.emplace_back(std::move(expr));
            }
            if (consumeToken(",").empty()) {
              if (consumeToken(")").empty()) {
                throw std::runtime_error("Expected closing parenthesis in call args");
              }
              return result;
            }
        }
        throw std::runtime_error("Expected closing parenthesis in call args");
    }

    std::shared_ptr<VariableExpr> parseIdentifier() {
        static std::regex ident_regex(R"((?!(?:not|is|and|or|del)\b)[a-zA-Z_]\w*)");
        auto location = get_location();
        auto ident = consumeToken(ident_regex);
        if (ident.empty())
          return nullptr;
        return std::make_shared<VariableExpr>(location, ident);
    }

    std::shared_ptr<Expression> parseStringConcat() {
        auto left = parseMathPow();
        if (!left) throw std::runtime_error("Expected left side of 'string concat' expression");

        static std::regex concat_tok(R"(~(?!\}))");
        if (!consumeToken(concat_tok).empty()) {
            auto right = parseLogicalAnd();
            if (!right) throw std::runtime_error("Expected right side of 'string concat' expression");
            left = std::make_shared<BinaryOpExpr>(get_location(), std::move(left), std::move(right), BinaryOpExpr::Op::StrConcat);
        }
        return left;
    }

    std::shared_ptr<Expression> parseMathPow() {
        auto left = parseMathPlusMinus();
        if (!left) throw std::runtime_error("Expected left side of 'math pow' expression");

        while (!consumeToken("**").empty()) {
            auto right = parseMathPlusMinus();
            if (!right) throw std::runtime_error("Expected right side of 'math pow' expression");
            left = std::make_shared<BinaryOpExpr>(get_location(), std::move(left), std::move(right), BinaryOpExpr::Op::MulMul);
        }
        return left;
    }

    std::shared_ptr<Expression> parseMathPlusMinus() {
        static std::regex plus_minus_tok(R"(\+|-(?![}%#]\}))");

        auto left = parseMathMulDiv();
        if (!left) throw std::runtime_error("Expected left side of 'math plus/minus' expression");
        std::string op_str;
        while (!(op_str = consumeToken(plus_minus_tok)).empty()) {
            auto right = parseMathMulDiv();
            if (!right) throw std::runtime_error("Expected right side of 'math plus/minus' expression");
            auto op = op_str == "+" ? BinaryOpExpr::Op::Add : BinaryOpExpr::Op::Sub;
            left = std::make_shared<BinaryOpExpr>(get_location(), std::move(left), std::move(right), op);
        }
        return left;
    }

    std::shared_ptr<Expression> parseMathMulDiv() {
        auto left = parseMathUnaryPlusMinus();
        if (!left) throw std::runtime_error("Expected left side of 'math mul/div' expression");

        static std::regex mul_div_tok(R"(\*\*?|//?|%(?!\}))");
        std::string op_str;
        while (!(op_str = consumeToken(mul_div_tok)).empty()) {
            auto right = parseMathUnaryPlusMinus();
            if (!right) throw std::runtime_error("Expected right side of 'math mul/div' expression");
            auto op = op_str == "*" ? BinaryOpExpr::Op::Mul
                : op_str == "**" ? BinaryOpExpr::Op::MulMul
                : op_str == "/" ? BinaryOpExpr::Op::Div
                : op_str == "//" ? BinaryOpExpr::Op::DivDiv
                : BinaryOpExpr::Op::Mod;
            left = std::make_shared<BinaryOpExpr>(get_location(), std::move(left), std::move(right), op);
        }

        if (!consumeToken("|").empty()) {
            auto expr = parseMathMulDiv();
            if (auto filter = dynamic_cast<FilterExpr*>(expr.get())) {
                filter->prepend(std::move(left));
                return expr;
            } else {
                std::vector<std::shared_ptr<Expression>> parts;
                parts.emplace_back(std::move(left));
                parts.emplace_back(std::move(expr));
                return std::make_shared<FilterExpr>(get_location(), std::move(parts));
            }
        }
        return left;
    }

    std::shared_ptr<Expression> call_func(const std::string & name, ArgumentsExpression && args) const {
        return std::make_shared<CallExpr>(get_location(), std::make_shared<VariableExpr>(get_location(), name), std::move(args));
    }

    std::shared_ptr<Expression> parseMathUnaryPlusMinus() {
        static std::regex unary_plus_minus_tok(R"(\+|-(?![}%#]\}))");
        auto op_str = consumeToken(unary_plus_minus_tok);
        auto expr = parseExpansion();
        if (!expr) throw std::runtime_error("Expected expr of 'unary plus/minus/expansion' expression");

        if (!op_str.empty()) {
            auto op = op_str == "+" ? UnaryOpExpr::Op::Plus : UnaryOpExpr::Op::Minus;
            return std::make_shared<UnaryOpExpr>(get_location(), std::move(expr), op);
        }
        return expr;
    }

    std::shared_ptr<Expression> parseExpansion() {
      static std::regex expansion_tok(R"(\*\*?)");
      auto op_str = consumeToken(expansion_tok);
      auto expr = parseValueExpression();
      if (op_str.empty()) return expr;
      if (!expr) throw std::runtime_error("Expected expr of 'expansion' expression");
      return std::make_shared<UnaryOpExpr>(get_location(), std::move(expr), op_str == "*" ? UnaryOpExpr::Op::Expansion : UnaryOpExpr::Op::ExpansionDict);
    }

    std::shared_ptr<Expression> parseValueExpression() {
      auto parseValue = [&]() -> std::shared_ptr<Expression> {
        auto location = get_location();
        auto constant = parseConstant();
        if (constant) return std::make_shared<LiteralExpr>(location, *constant);

        static std::regex null_regex(R"(null\b)");
        if (!consumeToken(null_regex).empty()) return std::make_shared<LiteralExpr>(location, Value());

        auto identifier = parseIdentifier();
        if (identifier) return identifier;

        auto braced = parseBracedExpressionOrArray();
        if (braced) return braced;

        auto array = parseArray();
        if (array) return array;

        auto dictionary = parseDictionary();
        if (dictionary) return dictionary;

        throw std::runtime_error("Expected value expression");
      };

      auto value = parseValue();

      while (it != end && consumeSpaces() && peekSymbols({ "[", ".", "(" })) {
        if (!consumeToken("[").empty()) {
          std::shared_ptr<Expression> index;
          auto slice_loc = get_location();
          std::shared_ptr<Expression> start, end, step;
          bool has_first_colon = false, has_second_colon = false;

          if (!peekSymbols({ ":" })) {
            start = parseExpression();
          }

          if (!consumeToken(":").empty()) {
            has_first_colon = true;
            if (!peekSymbols({ ":", "]" })) {
              end = parseExpression();
            }
            if (!consumeToken(":").empty()) {
              has_second_colon = true;
              if (!peekSymbols({ "]" })) {
                step = parseExpression();
              }
            }
          }

          if ((has_first_colon || has_second_colon)) {
            index = std::make_shared<SliceExpr>(slice_loc, std::move(start), std::move(end), std::move(step));
          } else {
            index = std::move(start);
          }
          if (!index) throw std::runtime_error("Empty index in subscript");
          if (consumeToken("]").empty()) throw std::runtime_error("Expected closing bracket in subscript");

          value = std::make_shared<SubscriptExpr>(value->location, std::move(value), std::move(index));
        } else if (!consumeToken(".").empty()) {
            auto identifier = parseIdentifier();
            if (!identifier) throw std::runtime_error("Expected identifier in subscript");

            consumeSpaces();
            if (peekSymbols({ "(" })) {
              auto callParams = parseCallArgs();
              value = std::make_shared<MethodCallExpr>(identifier->location, std::move(value), std::move(identifier), std::move(callParams));
            } else {
              auto key = std::make_shared<LiteralExpr>(identifier->location, Value(identifier->get_name()));
              value = std::make_shared<SubscriptExpr>(identifier->location, std::move(value), std::move(key));
            }
        } else if (peekSymbols({ "(" })) {
          auto callParams = parseCallArgs();
          value = std::make_shared<CallExpr>(get_location(), std::move(value), std::move(callParams));
        }
        consumeSpaces();
      }

      return value;
    }

    std::shared_ptr<Expression> parseBracedExpressionOrArray() {
        if (consumeToken("(").empty()) return nullptr;

        auto expr = parseExpression();
        if (!expr) throw std::runtime_error("Expected expression in braced expression");

        if (!consumeToken(")").empty()) {
            return expr;  // Drop the parentheses
        }

        std::vector<std::shared_ptr<Expression>> tuple;
        tuple.emplace_back(std::move(expr));

        while (it != end) {
          if (consumeToken(",").empty()) throw std::runtime_error("Expected comma in tuple");
          auto next = parseExpression();
          if (!next) throw std::runtime_error("Expected expression in tuple");
          tuple.push_back(std::move(next));

          if (!consumeToken(")").empty()) {
              return std::make_shared<ArrayExpr>(get_location(), std::move(tuple));
          }
        }
        throw std::runtime_error("Expected closing parenthesis");
    }

    std::shared_ptr<Expression> parseArray() {
        if (consumeToken("[").empty()) return nullptr;

        std::vector<std::shared_ptr<Expression>> elements;
        if (!consumeToken("]").empty()) {
            return std::make_shared<ArrayExpr>(get_location(), std::move(elements));
        }
        auto first_expr = parseExpression();
        if (!first_expr) throw std::runtime_error("Expected first expression in array");
        elements.push_back(std::move(first_expr));

        while (it != end) {
            if (!consumeToken(",").empty()) {
              auto expr = parseExpression();
              if (!expr) throw std::runtime_error("Expected expression in array");
              elements.push_back(std::move(expr));
            } else if (!consumeToken("]").empty()) {
                return std::make_shared<ArrayExpr>(get_location(), std::move(elements));
            } else {
                throw std::runtime_error("Expected comma or closing bracket in array");
            }
        }
        throw std::runtime_error("Expected closing bracket");
    }

    std::shared_ptr<Expression> parseDictionary() {
        if (consumeToken("{").empty()) return nullptr;

        std::vector<std::pair<std::shared_ptr<Expression>, std::shared_ptr<Expression>>> elements;
        if (!consumeToken("}").empty()) {
            return std::make_shared<DictExpr>(get_location(), std::move(elements));
        }

        auto parseKeyValuePair = [&]() {
            auto key = parseExpression();
            if (!key) throw std::runtime_error("Expected key in dictionary");
            if (consumeToken(":").empty()) throw std::runtime_error("Expected colon betweek key & value in dictionary");
            auto value = parseExpression();
            if (!value) throw std::runtime_error("Expected value in dictionary");
            elements.emplace_back(std::pair(std::move(key), std::move(value)));
        };

        parseKeyValuePair();

        while (it != end) {
            if (!consumeToken(",").empty()) {
                parseKeyValuePair();
            } else if (!consumeToken("}").empty()) {
                return std::make_shared<DictExpr>(get_location(), std::move(elements));
            } else {
                throw std::runtime_error("Expected comma or closing brace in dictionary");
            }
        }
        throw std::runtime_error("Expected closing brace");
    }

    SpaceHandling parsePreSpace(const std::string& s) const {
        if (s == "-")
          return SpaceHandling::Strip;
        return SpaceHandling::Keep;
    }

    SpaceHandling parsePostSpace(const std::string& s) const {
        if (s == "-") return SpaceHandling::Strip;
        return SpaceHandling::Keep;
    }

    using TemplateTokenVector = std::vector<std::unique_ptr<TemplateToken>>;
    using TemplateTokenIterator = TemplateTokenVector::const_iterator;

    std::vector<std::string> parseVarNames() {
      static std::regex varnames_regex(R"(((?:\w+)(?:\s*,\s*(?:\w+))*)\s*)");

      std::vector<std::string> group;
      if ((group = consumeTokenGroups(varnames_regex)).empty()) throw std::runtime_error("Expected variable names");
      std::vector<std::string> varnames;
      std::istringstream iss(group[1]);
      std::string varname;
      while (std::getline(iss, varname, ',')) {
        varnames.push_back(strip(varname));
      }
      return varnames;
    }

    std::runtime_error unexpected(const TemplateToken & token) const {
      return std::runtime_error("Unexpected " + TemplateToken::typeToString(token.type)
        + error_location_suffix(*template_str, token.location.pos));
    }
    std::runtime_error unterminated(const TemplateToken & token) const {
      return std::runtime_error("Unterminated " + TemplateToken::typeToString(token.type)
        + error_location_suffix(*template_str, token.location.pos));
    }

    TemplateTokenVector tokenize() {
      static std::regex comment_tok(R"(\{#([-~]?)([\s\S]*?)([-~]?)#\})");
      static std::regex expr_open_regex(R"(\{\{([-~])?)");
      static std::regex block_open_regex(R"(^\{%([-~])?\s*)");
      static std::regex block_keyword_tok(R"((if|else|elif|endif|for|endfor|generation|endgeneration|set|endset|block|endblock|macro|endmacro|filter|endfilter|break|continue|call|endcall)\b)");
      static std::regex non_text_open_regex(R"(\{\{|\{%|\{#)");
      static std::regex expr_close_regex(R"(\s*([-~])?\}\})");
      static std::regex block_close_regex(R"(\s*([-~])?%\})");

      TemplateTokenVector tokens;
      std::vector<std::string> group;
      std::string text;
      std::smatch match;

      try {
        while (it != end) {
          auto location = get_location();

          if (!(group = consumeTokenGroups(comment_tok, SpaceHandling::Keep)).empty()) {
            auto pre_space = parsePreSpace(group[1]);
            auto content = group[2];
            auto post_space = parsePostSpace(group[3]);
            tokens.push_back(std::make_unique<CommentTemplateToken>(location, pre_space, post_space, content));
          } else if (!(group = consumeTokenGroups(expr_open_regex, SpaceHandling::Keep)).empty()) {
            auto pre_space = parsePreSpace(group[1]);
            auto expr = parseExpression();

            if ((group = consumeTokenGroups(expr_close_regex)).empty()) {
              throw std::runtime_error("Expected closing expression tag");
            }

            auto post_space = parsePostSpace(group[1]);
            tokens.push_back(std::make_unique<ExpressionTemplateToken>(location, pre_space, post_space, std::move(expr)));
          } else if (!(group = consumeTokenGroups(block_open_regex, SpaceHandling::Keep)).empty()) {
            auto pre_space = parsePreSpace(group[1]);

            std::string keyword;

            auto parseBlockClose = [&]() -> SpaceHandling {
              if ((group = consumeTokenGroups(block_close_regex)).empty()) throw std::runtime_error("Expected closing block tag");
              return parsePostSpace(group[1]);
            };

            if ((keyword = consumeToken(block_keyword_tok)).empty()) throw std::runtime_error("Expected block keyword");

            if (keyword == "if") {
              auto condition = parseExpression();
              if (!condition) throw std::runtime_error("Expected condition in if block");

              auto post_space = parseBlockClose();
              tokens.push_back(std::make_unique<IfTemplateToken>(location, pre_space, post_space, std::move(condition)));
            } else if (keyword == "elif") {
              auto condition = parseExpression();
              if (!condition) throw std::runtime_error("Expected condition in elif block");

              auto post_space = parseBlockClose();
              tokens.push_back(std::make_unique<ElifTemplateToken>(location, pre_space, post_space, std::move(condition)));
            } else if (keyword == "else") {
              auto post_space = parseBlockClose();
              tokens.push_back(std::make_unique<ElseTemplateToken>(location, pre_space, post_space));
            } else if (keyword == "endif") {
              auto post_space = parseBlockClose();
              tokens.push_back(std::make_unique<EndIfTemplateToken>(location, pre_space, post_space));
            } else if (keyword == "for") {
              static std::regex recursive_tok(R"(recursive\b)");
              static std::regex if_tok(R"(if\b)");

              auto varnames = parseVarNames();
              static std::regex in_tok(R"(in\b)");
              if (consumeToken(in_tok).empty()) throw std::runtime_error("Expected 'in' keyword in for block");
              auto iterable = parseExpression(/* allow_if_expr = */ false);
              if (!iterable) throw std::runtime_error("Expected iterable in for block");

              std::shared_ptr<Expression> condition;
              if (!consumeToken(if_tok).empty()) {
                condition = parseExpression();
              }
              auto recursive = !consumeToken(recursive_tok).empty();

              auto post_space = parseBlockClose();
              tokens.push_back(std::make_unique<ForTemplateToken>(location, pre_space, post_space, std::move(varnames), std::move(iterable), std::move(condition), recursive));
            } else if (keyword == "endfor") {
              auto post_space = parseBlockClose();
              tokens.push_back(std::make_unique<EndForTemplateToken>(location, pre_space, post_space));
            } else if (keyword == "generation") {
              auto post_space = parseBlockClose();
              tokens.push_back(std::make_unique<GenerationTemplateToken>(location, pre_space, post_space));
            } else if (keyword == "endgeneration") {
              auto post_space = parseBlockClose();
              tokens.push_back(std::make_unique<EndGenerationTemplateToken>(location, pre_space, post_space));
            } else if (keyword == "set") {
              static std::regex namespaced_var_regex(R"((\w+)\s*\.\s*(\w+))");

              std::string ns;
              std::vector<std::string> var_names;
              std::shared_ptr<Expression> value;
              if (!(group = consumeTokenGroups(namespaced_var_regex)).empty()) {
                ns = group[1];
                var_names.push_back(group[2]);

                if (consumeToken("=").empty()) throw std::runtime_error("Expected equals sign in set block");

                value = parseExpression();
                if (!value) throw std::runtime_error("Expected value in set block");
              } else {
                var_names = parseVarNames();

                if (!consumeToken("=").empty()) {
                  value = parseExpression();
                  if (!value) throw std::runtime_error("Expected value in set block");
                }
              }
              auto post_space = parseBlockClose();
              tokens.push_back(std::make_unique<SetTemplateToken>(location, pre_space, post_space, ns, var_names, std::move(value)));
            } else if (keyword == "endset") {
              auto post_space = parseBlockClose();
              tokens.push_back(std::make_unique<EndSetTemplateToken>(location, pre_space, post_space));
            } else if (keyword == "macro") {
              auto macroname = parseIdentifier();
              if (!macroname) throw std::runtime_error("Expected macro name in macro block");
              auto params = parseParameters();

              auto post_space = parseBlockClose();
              tokens.push_back(std::make_unique<MacroTemplateToken>(location, pre_space, post_space, std::move(macroname), std::move(params)));
            } else if (keyword == "endmacro") {
              auto post_space = parseBlockClose();
              tokens.push_back(std::make_unique<EndMacroTemplateToken>(location, pre_space, post_space));
            } else if (keyword == "call") {
              auto expr = parseExpression();
              if (!expr) throw std::runtime_error("Expected expression in call block");

              auto post_space = parseBlockClose();
              tokens.push_back(std::make_unique<CallTemplateToken>(location, pre_space, post_space, std::move(expr)));
            } else if (keyword == "endcall") {
              auto post_space = parseBlockClose();
              tokens.push_back(std::make_unique<EndCallTemplateToken>(location, pre_space, post_space));
            } else if (keyword == "filter") {
              auto filter = parseExpression();
              if (!filter) throw std::runtime_error("Expected expression in filter block");

              auto post_space = parseBlockClose();
              tokens.push_back(std::make_unique<FilterTemplateToken>(location, pre_space, post_space, std::move(filter)));
            } else if (keyword == "endfilter") {
              auto post_space = parseBlockClose();
              tokens.push_back(std::make_unique<EndFilterTemplateToken>(location, pre_space, post_space));
            } else if (keyword == "break" || keyword == "continue") {
              auto post_space = parseBlockClose();
              tokens.push_back(std::make_unique<LoopControlTemplateToken>(location, pre_space, post_space, keyword == "break" ? LoopControlType::Break : LoopControlType::Continue));
            } else {
              throw std::runtime_error("Unexpected block: " + keyword);
            }
          } else if (std::regex_search(it, end, match, non_text_open_regex)) {
            if (!match.position()) {
                if (match[0] != "{#")
                    throw std::runtime_error("Internal error: Expected a comment");
                throw std::runtime_error("Missing end of comment tag");
            }
            auto text_end = it + match.position();
            text = std::string(it, text_end);
            it = text_end;
            tokens.push_back(std::make_unique<TextTemplateToken>(location, SpaceHandling::Keep, SpaceHandling::Keep, text));
          } else {
            text = std::string(it, end);
            it = end;
            tokens.push_back(std::make_unique<TextTemplateToken>(location, SpaceHandling::Keep, SpaceHandling::Keep, text));
          }
        }
        return tokens;
      } catch (const std::exception & e) {
        throw std::runtime_error(e.what() + error_location_suffix(*template_str, std::distance(start, it)));
      }
    }

    std::shared_ptr<TemplateNode> parseTemplate(
          const TemplateTokenIterator & begin,
          TemplateTokenIterator & it,
          const TemplateTokenIterator & end,
          bool fully = false) const {
        std::vector<std::shared_ptr<TemplateNode>> children;
        while (it != end) {
          const auto start = it;
          const auto & token = *(it++);
          if (auto if_token = dynamic_cast<IfTemplateToken*>(token.get())) {
              std::vector<std::pair<std::shared_ptr<Expression>, std::shared_ptr<TemplateNode>>> cascade;
              cascade.emplace_back(std::move(if_token->condition), parseTemplate(begin, it, end));

              while (it != end && (*it)->type == TemplateToken::Type::Elif) {
                  auto elif_token = dynamic_cast<ElifTemplateToken*>((*(it++)).get());
                  cascade.emplace_back(std::move(elif_token->condition), parseTemplate(begin, it, end));
              }

              if (it != end && (*it)->type == TemplateToken::Type::Else) {
                cascade.emplace_back(nullptr, parseTemplate(begin, ++it, end));
              }
              if (it == end || (*(it++))->type != TemplateToken::Type::EndIf) {
                  throw unterminated(**start);
              }
              children.emplace_back(std::make_shared<IfNode>(token->location, std::move(cascade)));
          } else if (auto for_token = dynamic_cast<ForTemplateToken*>(token.get())) {
              auto body = parseTemplate(begin, it, end);
              auto else_body = std::shared_ptr<TemplateNode>();
              if (it != end && (*it)->type == TemplateToken::Type::Else) {
                else_body = parseTemplate(begin, ++it, end);
              }
              if (it == end || (*(it++))->type != TemplateToken::Type::EndFor) {
                  throw unterminated(**start);
              }
              children.emplace_back(std::make_shared<ForNode>(token->location, std::move(for_token->var_names), std::move(for_token->iterable), std::move(for_token->condition), std::move(body), for_token->recursive, std::move(else_body)));
          } else if (dynamic_cast<GenerationTemplateToken*>(token.get())) {
              auto body = parseTemplate(begin, it, end);
              if (it == end || (*(it++))->type != TemplateToken::Type::EndGeneration) {
                  throw unterminated(**start);
              }
              // Treat as a no-op, as our scope is templates for inference, not training (`{% generation %}` wraps generated tokens for masking).
              children.emplace_back(std::move(body));
          } else if (auto text_token = dynamic_cast<TextTemplateToken*>(token.get())) {
              SpaceHandling pre_space = (it - 1) != begin ? (*(it - 2))->post_space : SpaceHandling::Keep;
              SpaceHandling post_space = it != end ? (*it)->pre_space : SpaceHandling::Keep;

              auto text = text_token->text;
              if (post_space == SpaceHandling::Strip) {
                static std::regex trailing_space_regex(R"(\s+$)");
                text = std::regex_replace(text, trailing_space_regex, "");
              } else if (options.lstrip_blocks && it != end) {
                auto i = text.size();
                while (i > 0 && (text[i - 1] == ' ' || text[i - 1] == '\t')) i--;
                if ((i == 0 && (it - 1) == begin) || (i > 0 && text[i - 1] == '\n')) {
                  text.resize(i);
                }
              }
              if (pre_space == SpaceHandling::Strip) {
                static std::regex leading_space_regex(R"(^\s+)");
                text = std::regex_replace(text, leading_space_regex, "");
              } else if (options.trim_blocks && (it - 1) != begin && !dynamic_cast<ExpressionTemplateToken*>((*(it - 2)).get())) {
                if (!text.empty() && text[0] == '\n') {
                  text.erase(0, 1);
                }
              }
              if (it == end && !options.keep_trailing_newline) {
                auto i = text.size();
                if (i > 0 && text[i - 1] == '\n') {
                  i--;
                  if (i > 0 && text[i - 1] == '\r') i--;
                  text.resize(i);
                }
              }
              children.emplace_back(std::make_shared<TextNode>(token->location, text));
          } else if (auto expr_token = dynamic_cast<ExpressionTemplateToken*>(token.get())) {
              children.emplace_back(std::make_shared<ExpressionNode>(token->location, std::move(expr_token->expr)));
          } else if (auto set_token = dynamic_cast<SetTemplateToken*>(token.get())) {
            if (set_token->value) {
              children.emplace_back(std::make_shared<SetNode>(token->location, set_token->ns, set_token->var_names, std::move(set_token->value)));
            } else {
              auto value_template = parseTemplate(begin, it, end);
              if (it == end || (*(it++))->type != TemplateToken::Type::EndSet) {
                  throw unterminated(**start);
              }
              if (!set_token->ns.empty()) throw std::runtime_error("Namespaced set not supported in set with template value");
              if (set_token->var_names.size() != 1) throw std::runtime_error("Structural assignment not supported in set with template value");
              auto & name = set_token->var_names[0];
              children.emplace_back(std::make_shared<SetTemplateNode>(token->location, name, std::move(value_template)));
            }
          } else if (auto macro_token = dynamic_cast<MacroTemplateToken*>(token.get())) {
              auto body = parseTemplate(begin, it, end);
              if (it == end || (*(it++))->type != TemplateToken::Type::EndMacro) {
                  throw unterminated(**start);
              }
              children.emplace_back(std::make_shared<MacroNode>(token->location, std::move(macro_token->name), std::move(macro_token->params), std::move(body)));
          } else if (auto call_token = dynamic_cast<CallTemplateToken*>(token.get())) {
            auto body = parseTemplate(begin, it, end);
            if (it == end || (*(it++))->type != TemplateToken::Type::EndCall) {
                throw unterminated(**start);
            }
            children.emplace_back(std::make_shared<CallNode>(token->location, std::move(call_token->expr), std::move(body)));
          } else if (auto filter_token = dynamic_cast<FilterTemplateToken*>(token.get())) {
              auto body = parseTemplate(begin, it, end);
              if (it == end || (*(it++))->type != TemplateToken::Type::EndFilter) {
                  throw unterminated(**start);
              }
              children.emplace_back(std::make_shared<FilterNode>(token->location, std::move(filter_token->filter), std::move(body)));
          } else if (dynamic_cast<CommentTemplateToken*>(token.get())) {
              // Ignore comments
          } else if (auto ctrl_token = dynamic_cast<LoopControlTemplateToken*>(token.get())) {
              children.emplace_back(std::make_shared<LoopControlNode>(token->location, ctrl_token->control_type));
          } else if (dynamic_cast<EndForTemplateToken*>(token.get())
                  || dynamic_cast<EndSetTemplateToken*>(token.get())
                  || dynamic_cast<EndMacroTemplateToken*>(token.get())
                  || dynamic_cast<EndCallTemplateToken*>(token.get())
                  || dynamic_cast<EndFilterTemplateToken*>(token.get())
                  || dynamic_cast<EndIfTemplateToken*>(token.get())
                  || dynamic_cast<ElseTemplateToken*>(token.get())
                  || dynamic_cast<EndGenerationTemplateToken*>(token.get())
                  || dynamic_cast<ElifTemplateToken*>(token.get())) {
              it--;  // unconsume the token
              break;  // exit the loop
          } else {
              throw unexpected(**(it-1));
          }
        }
        if (fully && it != end) {
            throw unexpected(**it);
        }
        if (children.empty()) {
          return std::make_shared<TextNode>(Location { template_str, 0 }, std::string());
        } else if (children.size() == 1) {
          return std::move(children[0]);
        } else {
          return std::make_shared<SequenceNode>(children[0]->location(), std::move(children));
        }
    }

public:

    static std::shared_ptr<TemplateNode> parse(const std::string& template_str, const Options & options) {
        Parser parser(std::make_shared<std::string>(normalize_newlines(template_str)), options);
        auto tokens = parser.tokenize();
        TemplateTokenIterator begin = tokens.begin();
        auto it = begin;
        TemplateTokenIterator end = tokens.end();
        return parser.parseTemplate(begin, it, end, /* fully= */ true);
    }
};

static Value simple_function(const std::string & fn_name, const std::vector<std::string> & params, const std::function<Value(const std::shared_ptr<Context> &, Value & args)> & fn) {
  std::map<std::string, size_t> named_positions;
  for (size_t i = 0, n = params.size(); i < n; i++) named_positions[params[i]] = i;

  return Value::callable([=](const std::shared_ptr<Context> & context, ArgumentsValue & args) -> Value {
    auto args_obj = Value::object();
    std::vector<bool> provided_args(params.size());
    for (size_t i = 0, n = args.args.size(); i < n; i++) {
      auto & arg = args.args[i];
      if (i < params.size()) {
        args_obj.set(params[i], arg);
        provided_args[i] = true;
      } else {
        throw std::runtime_error("Too many positional params for " + fn_name);
      }
    }
    for (auto & [name, value] : args.kwargs) {
      auto named_pos_it = named_positions.find(name);
      if (named_pos_it == named_positions.end()) {
        throw std::runtime_error("Unknown argument " + name + " for function " + fn_name);
      }
      provided_args[named_pos_it->second] = true;
      args_obj.set(name, value);
    }
    return fn(context, args_obj);
  });
}

inline std::shared_ptr<Context> Context::builtins() {
  auto globals = Value::object();

  globals.set("raise_exception", simple_function("raise_exception", { "message" }, [](const std::shared_ptr<Context> &, Value & args) -> Value {
    throw std::runtime_error(args.at("message").get<std::string>());
  }));
  globals.set("tojson", simple_function("tojson", { "value", "indent", "ensure_ascii" }, [](const std::shared_ptr<Context> &, Value & args) {
    return Value(args.at("value").dump(args.get<int64_t>("indent", -1), /* to_json= */ true));
  }));
  globals.set("items", simple_function("items", { "object" }, [](const std::shared_ptr<Context> &, Value & args) {
    auto items = Value::array();
    if (args.contains("object")) {
      auto & obj = args.at("object");
      if (!obj.is_object()) {
        throw std::runtime_error("Can only get item pairs from a mapping");
      }
      for (auto & key : obj.keys()) {
        items.push_back(Value::array({key, obj.at(key)}));
      }
    }
    return items;
  }));
  globals.set("last", simple_function("last", { "items" }, [](const std::shared_ptr<Context> &, Value & args) {
    auto items = args.at("items");
    if (!items.is_array()) throw std::runtime_error("object is not a list");
    if (items.empty()) return Value();
    return items.at(items.size() - 1);
  }));
  globals.set("trim", simple_function("trim", { "text" }, [](const std::shared_ptr<Context> &, Value & args) {
    auto & text = args.at("text");
    return text.is_null() ? text : Value(strip(text.get<std::string>()));
  }));
  auto char_transform_function = [](const std::string & name, const std::function<char(char)> & fn) {
    return simple_function(name, { "text" }, [=](const std::shared_ptr<Context> &, Value & args) {
      auto text = args.at("text");
      if (text.is_null()) return text;
      std::string res;
      auto str = text.get<std::string>();
      std::transform(str.begin(), str.end(), std::back_inserter(res), fn);
      return Value(res);
    });
  };
  globals.set("lower", char_transform_function("lower", ::tolower));
  globals.set("upper", char_transform_function("upper", ::toupper));
  globals.set("default", Value::callable([=](const std::shared_ptr<Context> &, ArgumentsValue & args) {
    args.expectArgs("default", {2, 3}, {0, 1});
    auto & value = args.args[0];
    auto & default_value = args.args[1];
    bool boolean = false;
    if (args.args.size() == 3) {
      boolean = args.args[2].get<bool>();
    } else {
      Value bv = args.get_named("boolean");
      if (!bv.is_null()) {
        boolean = bv.get<bool>();
      }
    }
    return boolean ? (value.to_bool() ? value : default_value) : value.is_null() ? default_value : value;
  }));
  auto escape = simple_function("escape", { "text" }, [](const std::shared_ptr<Context> &, Value & args) {
    return Value(html_escape(args.at("text").get<std::string>()));
  });
  globals.set("e", escape);
  globals.set("escape", escape);
  globals.set("joiner", simple_function("joiner", { "sep" }, [](const std::shared_ptr<Context> &, Value & args) {
    auto sep = args.get<std::string>("sep", "");
    auto first = std::make_shared<bool>(true);
    return simple_function("", {}, [sep, first](const std::shared_ptr<Context> &, const Value &) -> Value {
      if (*first) {
        *first = false;
        return "";
      }
      return sep;
    });
    return Value(html_escape(args.at("text").get<std::string>()));
  }));
  globals.set("count", simple_function("count", { "items" }, [](const std::shared_ptr<Context> &, Value & args) {
    return Value((int64_t) args.at("items").size());
  }));
  globals.set("dictsort", simple_function("dictsort", { "value" }, [](const std::shared_ptr<Context> &, Value & args) {
    if (args.size() != 1) throw std::runtime_error("dictsort expects exactly 1 argument (TODO: fix implementation)");
    auto & value = args.at("value");
    auto keys = value.keys();
    std::sort(keys.begin(), keys.end());
    auto res = Value::array();
    for (auto & key : keys) {
      res.push_back(Value::array({key, value.at(key)}));
    }
    return res;
  }));
  globals.set("join", simple_function("join", { "items", "d" }, [](const std::shared_ptr<Context> &, Value & args) {
    auto do_join = [](Value & items, const std::string & sep) {
      if (!items.is_array()) throw std::runtime_error("object is not iterable: " + items.dump());
      std::ostringstream oss;
      auto first = true;
      for (size_t i = 0, n = items.size(); i < n; ++i) {
        if (first) first = false;
        else oss << sep;
        oss << items.at(i).to_str();
      }
      return Value(oss.str());
    };
    auto sep = args.get<std::string>("d", "");
    if (args.contains("items")) {
        auto & items = args.at("items");
        return do_join(items, sep);
    } else {
      return simple_function("", {"items"}, [sep, do_join](const std::shared_ptr<Context> &, Value & args) {
        auto & items = args.at("items");
        if (!items.to_bool() || !items.is_array()) throw std::runtime_error("join expects an array for items, got: " + items.dump());
        return do_join(items, sep);
      });
    }
  }));
  globals.set("namespace", Value::callable([=](const std::shared_ptr<Context> &, ArgumentsValue & args) {
    auto ns = Value::object();
    args.expectArgs("namespace", {0, 0}, {0, (std::numeric_limits<size_t>::max)()});
    for (auto & [name, value] : args.kwargs) {
      ns.set(name, value);
    }
    return ns;
  }));
  auto equalto = simple_function("equalto", { "expected", "actual" }, [](const std::shared_ptr<Context> &, Value & args) -> Value {
      return args.at("actual") == args.at("expected");
  });
  globals.set("equalto", equalto);
  globals.set("==", equalto);
  globals.set("length", simple_function("length", { "items" }, [](const std::shared_ptr<Context> &, Value & args) -> Value {
      auto & items = args.at("items");
      return (int64_t) items.size();
  }));
  globals.set("safe", simple_function("safe", { "value" }, [](const std::shared_ptr<Context> &, Value & args) -> Value {
      return args.at("value").to_str();
  }));
  globals.set("string", simple_function("string", { "value" }, [](const std::shared_ptr<Context> &, Value & args) -> Value {
      return args.at("value").to_str();
  }));
  globals.set("int", simple_function("int", { "value" }, [](const std::shared_ptr<Context> &, Value & args) -> Value {
      return args.at("value").to_int();
  }));
  globals.set("list", simple_function("list", { "items" }, [](const std::shared_ptr<Context> &, Value & args) -> Value {
      auto & items = args.at("items");
      if (!items.is_array()) throw std::runtime_error("object is not iterable");
      return items;
  }));
  globals.set("in", simple_function("in", { "item", "items" }, [](const std::shared_ptr<Context> &, Value & args) -> Value {
      return in(args.at("item"), args.at("items"));
  }));
  globals.set("unique", simple_function("unique", { "items" }, [](const std::shared_ptr<Context> &, Value & args) -> Value {
      auto & items = args.at("items");
      if (!items.is_array()) throw std::runtime_error("object is not iterable");
      std::unordered_set<Value> seen;
      auto result = Value::array();
      for (size_t i = 0, n = items.size(); i < n; i++) {
        auto pair = seen.insert(items.at(i));
        if (pair.second) {
          result.push_back(items.at(i));
        }
      }
      return result;
  }));
  auto make_filter = [](const Value & filter, Value & extra_args) -> Value {
    return simple_function("", { "value" }, [=](const std::shared_ptr<Context> & context, Value & args) {
      auto & value = args.at("value");
      ArgumentsValue actual_args;
      actual_args.args.emplace_back(value);
      for (size_t i = 0, n = extra_args.size(); i < n; i++) {
        actual_args.args.emplace_back(extra_args.at(i));
      }
      return filter.call(context, actual_args);
    });
  };
  auto select_or_reject = [make_filter](bool is_select) {
    return Value::callable([=](const std::shared_ptr<Context> & context, ArgumentsValue & args) {
      args.expectArgs(is_select ? "select" : "reject", {2, (std::numeric_limits<size_t>::max)()}, {0, 0});
      auto & items = args.args[0];
      if (items.is_null()) {
        return Value::array();
      }
      if (!items.is_array()) {
        throw std::runtime_error("object is not iterable: " + items.dump());
      }

      auto filter_fn = context->get(args.args[1]);
      if (filter_fn.is_null()) {
        throw std::runtime_error("Undefined filter: " + args.args[1].dump());
      }

      auto filter_args = Value::array();
      for (size_t i = 2, n = args.args.size(); i < n; i++) {
        filter_args.push_back(args.args[i]);
      }
      auto filter = make_filter(filter_fn, filter_args);

      auto res = Value::array();
      for (size_t i = 0, n = items.size(); i < n; i++) {
        auto & item = items.at(i);
        ArgumentsValue filter_args;
        filter_args.args.emplace_back(item);
        auto pred_res = filter.call(context, filter_args);
        if (pred_res.to_bool() == (is_select ? true : false)) {
          res.push_back(item);
        }
      }
      return res;
    });
  };
  globals.set("select", select_or_reject(/* is_select= */ true));
  globals.set("reject", select_or_reject(/* is_select= */ false));
  globals.set("map", Value::callable([=](const std::shared_ptr<Context> & context, ArgumentsValue & args) {
    auto res = Value::array();
    if (args.args.size() == 1 &&
      ((args.has_named("attribute") && args.kwargs.size() == 1) || (args.has_named("default") && args.kwargs.size() == 2))) {
      auto & items = args.args[0];
      auto attr_name = args.get_named("attribute");
      auto default_value = args.get_named("default");
      for (size_t i = 0, n = items.size(); i < n; i++) {
        auto & item = items.at(i);
        auto attr = item.get(attr_name);
        res.push_back(attr.is_null() ? default_value : attr);
      }
    } else if (args.kwargs.empty() && args.args.size() >= 2) {
      auto fn = context->get(args.args[1]);
      if (fn.is_null()) throw std::runtime_error("Undefined filter: " + args.args[1].dump());
      ArgumentsValue filter_args { {Value()}, {} };
      for (size_t i = 2, n = args.args.size(); i < n; i++) {
        filter_args.args.emplace_back(args.args[i]);
      }
      for (size_t i = 0, n = args.args[0].size(); i < n; i++) {
        auto & item = args.args[0].at(i);
        filter_args.args[0] = item;
        res.push_back(fn.call(context, filter_args));
      }
    } else {
      throw std::runtime_error("Invalid or unsupported arguments for map");
    }
    return res;
  }));
  globals.set("indent", simple_function("indent", { "text", "indent", "first" }, [](const std::shared_ptr<Context> &, Value & args) {
    auto text = args.at("text").get<std::string>();
    auto first = args.get<bool>("first", false);
    std::string out;
    std::string indent(args.get<int64_t>("indent", 0), ' ');
    std::istringstream iss(text);
    std::string line;
    auto is_first = true;
    while (std::getline(iss, line, '\n')) {
      auto needs_indent = !is_first || first;
      if (is_first) is_first = false;
      else out += "\n";
      if (needs_indent) out += indent;
      out += line;
    }
    if (!text.empty() && text.back() == '\n') out += "\n";
    return out;
  }));
  auto select_or_reject_attr = [](bool is_select) {
    return Value::callable([=](const std::shared_ptr<Context> & context, ArgumentsValue & args) {
      args.expectArgs(is_select ? "selectattr" : "rejectattr", {2, (std::numeric_limits<size_t>::max)()}, {0, 0});
      auto & items = args.args[0];
      if (items.is_null())
        return Value::array();
      if (!items.is_array()) throw std::runtime_error("object is not iterable: " + items.dump());
      auto attr_name = args.args[1].get<std::string>();

      bool has_test = false;
      Value test_fn;
      ArgumentsValue test_args {{Value()}, {}};
      if (args.args.size() >= 3) {
        has_test = true;
        test_fn = context->get(args.args[2]);
        if (test_fn.is_null()) throw std::runtime_error("Undefined test: " + args.args[2].dump());
        for (size_t i = 3, n = args.args.size(); i < n; i++) {
          test_args.args.emplace_back(args.args[i]);
        }
        test_args.kwargs = args.kwargs;
      }

      auto res = Value::array();
      for (size_t i = 0, n = items.size(); i < n; i++) {
        auto & item = items.at(i);
        auto attr = item.get(attr_name);
        if (has_test) {
          test_args.args[0] = attr;
          if (test_fn.call(context, test_args).to_bool() == (is_select ? true : false)) {
            res.push_back(item);
          }
        } else {
          res.push_back(attr);
        }
      }
      return res;
    });
  };
  globals.set("selectattr", select_or_reject_attr(/* is_select= */ true));
  globals.set("rejectattr", select_or_reject_attr(/* is_select= */ false));
  globals.set("range", Value::callable([=](const std::shared_ptr<Context> &, ArgumentsValue & args) {
    std::vector<int64_t> startEndStep(3);
    std::vector<bool> param_set(3);
    if (args.args.size() == 1) {
      startEndStep[1] = args.args[0].get<int64_t>();
      param_set[1] = true;
    } else {
      for (size_t i = 0; i < args.args.size(); i++) {
        auto & arg = args.args[i];
        auto v = arg.get<int64_t>();
        startEndStep[i] = v;
        param_set[i] = true;
      }
    }
    for (auto & [name, value] : args.kwargs) {
      size_t i;
      if (name == "start") {
        i = 0;
      } else if (name == "end") {
        i = 1;
      } else if (name == "step") {
        i = 2;
      } else {
        throw std::runtime_error("Unknown argument " + name + " for function range");
      }

      if (param_set[i]) {
        throw std::runtime_error("Duplicate argument " + name + " for function range");
      }
      startEndStep[i] = value.get<int64_t>();
      param_set[i] = true;
    }
    if (!param_set[1]) {
      throw std::runtime_error("Missing required argument 'end' for function range");
    }
    int64_t start = param_set[0] ? startEndStep[0] : 0;
    int64_t end = startEndStep[1];
    int64_t step = param_set[2] ? startEndStep[2] : 1;

    auto res = Value::array();
    if (step > 0) {
      for (int64_t i = start; i < end; i += step) {
        res.push_back(Value(i));
      }
    } else {
      for (int64_t i = start; i > end; i += step) {
        res.push_back(Value(i));
      }
    }
    return res;
  }));

  return std::make_shared<Context>(std::move(globals));
}

inline std::shared_ptr<Context> Context::make(Value && values, const std::shared_ptr<Context> & parent) {
  return std::make_shared<Context>(values.is_null() ? Value::object() : std::move(values), parent);
}

}  // namespace minja
