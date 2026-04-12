#include "openvino/pass/matcher_pass.hpp"

namespace ov {
namespace frontend {
namespace ggml {
namespace pass {

class EliminateZeroPoints : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::ggml::pass::EliminateZeroPoints")
    EliminateZeroPoints();
};

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
