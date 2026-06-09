#include "renderer/core/render_pipeline.h"
#include <algorithm>

namespace somegi {

void RenderPipeline::addStep(RenderStep step) {
    m_index[step.name] = m_registry.size();
    m_registry.push_back(std::move(step));
}

void RenderPipeline::build() {
    m_execTable.clear();
    m_execTable.reserve(m_registry.size());
    for (auto& step : m_registry) {
        if (step.enabled) {
            m_execTable.push_back(&step);
        }
    }
}

void RenderPipeline::execute(VkCommandBuffer cmd) {
    for (auto* step : m_execTable) {
        if (step->record) {
            step->record(cmd);
        }
    }
}

void RenderPipeline::setEnabled(const std::string& name, bool enabled) {
    auto it = m_index.find(name);
    if (it != m_index.end()) {
        m_registry[it->second].enabled = enabled;
    }
}

void RenderPipeline::clear() {
    m_registry.clear();
    m_execTable.clear();
    m_index.clear();
}

} // namespace somegi
