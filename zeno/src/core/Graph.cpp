#include <zeno/core/Graph.h>
#ifdef ZENO_AETHER_ENABLE_PROFILER
#include <zeno/core/NodeExecutionProfiler.h>
#endif
#include <zeno/core/INode.h>
#include <zeno/core/IObject.h>
#include <zeno/core/Session.h>
#include <zeno/utils/safe_at.h>
#include <zeno/utils/scope_exit.h>
#include <zeno/core/Descriptor.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/extra/GraphException.h>
#include <zeno/funcs/LiterialConverter.h>
#include <zeno/extra/GlobalStatus.h>
#include <zeno/extra/SubnetNode.h>
#include <zeno/extra/DirtyChecker.h>
#include <zeno/utils/Error.h>
#include <zeno/utils/log.h>
#include <iostream>
#ifdef ZENO_AETHER_ENABLE_SMART_SCHEDULER
#include <algorithm>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#endif

namespace zeno {

ZENO_API Context::Context() = default;
ZENO_API Context::~Context() = default;

ZENO_API Context::Context(Context const &other)
    : visited(other.visited)
{}

ZENO_API Graph::Graph() = default;
ZENO_API Graph::~Graph() {
#ifdef ZENO_AETHER_ENABLE_SMART_SCHEDULER
    clearAetherNodeCostHistory(this);
#endif
}

ZENO_API zany const &Graph::getNodeOutput(
    std::string const &sn, std::string const &ss) const {
    auto node = safe_at(nodes, sn, "node name").get();
    if (node->muted_output)
        return node->muted_output;
    return safe_at(node->outputs, ss, "output socket name of node " + node->myname);
}

zany Graph::getNodeInput(std::string const& sn, std::string const& ss) const {
    auto node = safe_at(nodes, sn, "node name").get();
    return node->get_input(ss);
}

ZENO_API void Graph::clearNodes() {
    nodes.clear();
}

ZENO_API void Graph::addNode(std::string const &cls, std::string const &id) {
    if (nodes.find(id) != nodes.end())
        return;  // no add twice, to prevent output object invalid
    auto cl = safe_at(session->nodeClasses, cls, "node class name").get();
    auto node = cl->new_instance();
    node->graph = this;
    node->myname = id;
    node->nodeClass = cl;
    nodes[id] = std::move(node);
}

ZENO_API Graph *Graph::addSubnetNode(std::string const &id) {
    auto subcl = std::make_unique<ImplSubnetNodeClass>();
    auto node = subcl->new_instance();
    node->graph = this;
    node->myname = id;
    node->nodeClass = subcl.get();
    auto subnode = static_cast<SubnetNode *>(node.get());
    subnode->subgraph->session = this->session;
    subnode->subnetClass = std::move(subcl);
    auto subg = subnode->subgraph.get();
    nodes[id] = std::move(node);
    return subg;
}

ZENO_API Graph *Graph::getSubnetGraph(std::string const &id) const {
    auto node = static_cast<SubnetNode *>(safe_at(nodes, id, "node name").get());
    return node->subgraph.get();
}

ZENO_API void Graph::completeNode(std::string const &id) {
    safe_at(nodes, id, "node name")->doComplete();
}

ZENO_API bool Graph::applyNode(std::string const &id) {
    if (ctx->visited.find(id) != ctx->visited.end()) {
        return false;
    }
    ctx->visited.insert(id);
    auto node = safe_at(nodes, id, "node name").get();
    GraphException::translated([&] {
        node->doApply();
    }, node->myname);
    if (dirtyChecker && dirtyChecker->amIDirty(id)) {
        return true;
    }
    return false;
}

ZENO_API void Graph::applyNodes(std::set<std::string> const &ids) {
#ifdef ZENO_AETHER_ENABLE_PROFILER
    AetherExecutionProfileScope executionProfile;
#endif
    ctx = std::make_unique<Context>();

    scope_exit _{[&] {
        ctx = nullptr;
    }};

#ifdef ZENO_AETHER_ENABLE_SMART_SCHEDULER
    // Only top-level requested nodes are reordered. Dependencies still execute
    // through the existing recursive applyNode() path, preserving graph semantics.
    std::unordered_map<std::string, double> pathCostCache;
    std::unordered_map<std::string, std::uint64_t> sampleCache;
    std::unordered_set<std::string> visiting;
    std::function<double(const std::string &)> estimateCriticalPath =
        [&](const std::string &nodeId) -> double {
            const auto cached = pathCostCache.find(nodeId);
            if (cached != pathCostCache.end()) return cached->second;
            if (visiting.find(nodeId) != visiting.end()) return 0.0;

            const auto nodeIt = nodes.find(nodeId);
            if (nodeIt == nodes.end()) return 0.0;
            visiting.insert(nodeId);

            const auto estimate = getAetherNodeCostEstimate(nodeId, this);
            sampleCache[nodeId] = estimate.samples;
            double longestDependencyPath = 0.0;
            for (const auto &binding : nodeIt->second->inputBounds) {
                const auto dependencyId = binding.second.first;
                if (nodes.find(dependencyId) != nodes.end()) {
                    longestDependencyPath = std::max(
                        longestDependencyPath, estimateCriticalPath(dependencyId));
                }
            }

            visiting.erase(nodeId);
            const double confidenceWeightedNodeCost =
                estimate.meanApplyMilliseconds * estimate.confidence;
            const double pathCost =
                confidenceWeightedNodeCost + longestDependencyPath;
            pathCostCache[nodeId] = pathCost;
            return pathCost;
        };

    std::vector<AetherScheduleEntry> scheduledRoots;
    scheduledRoots.reserve(ids.size());
    for (const auto &id : ids) {
        AetherScheduleEntry entry;
        entry.node = id;
        entry.estimatedCriticalPathMilliseconds = estimateCriticalPath(id);
        entry.historicalSamples = sampleCache[id];
        entry.reason = entry.historicalSamples
            ? "ordered by confidence-weighted historical EWMA plus longest known dependency path"
            : "no root timing history; stable ties retain deterministic input order";
        scheduledRoots.emplace_back(std::move(entry));
    }

    std::stable_sort(scheduledRoots.begin(), scheduledRoots.end(),
        [](const AetherScheduleEntry &a, const AetherScheduleEntry &b) {
            return a.estimatedCriticalPathMilliseconds >
                   b.estimatedCriticalPathMilliseconds;
        });
    recordAetherScheduleOrder(scheduledRoots);
    for (const auto &entry : scheduledRoots) {
        applyNode(entry.node);
    }
#else
    for (auto const &id: ids) {
        applyNode(id);
    }
#endif
}

ZENO_API void Graph::applyNodesToExec() {
    log_debug("{} nodes to exec", nodesToExec.size());
    applyNodes(nodesToExec);
}

ZENO_API void Graph::bindNodeInput(std::string const &dn, std::string const &ds,
        std::string const &sn, std::string const &ss) {
    safe_at(nodes, dn, "node name")->inputBounds[ds] = std::pair(sn, ss);
}

ZENO_API void Graph::setNodeInput(std::string const &id, std::string const &par,
        zany const &val) {
    safe_at(nodes, id, "node name")->inputs[par] = val;
}

ZENO_API void Graph::setKeyFrame(std::string const &id, std::string const &par, zany const &val) {
    safe_at(nodes, id, "node name")->inputs[par] = val;
    safe_at(nodes, id, "node name")->kframes.insert(par);
}

ZENO_API void Graph::setFormula(std::string const &id, std::string const &par, zany const &val) {
    safe_at(nodes, id, "node name")->inputs[par] = val;
    safe_at(nodes, id, "node name")->formulas.insert(par);
}


ZENO_API std::map<std::string, zany> Graph::callSubnetNode(std::string const &id,
        std::map<std::string, zany> inputs) const {
    auto se = safe_at(nodes, id, "node name").get();
    se->inputs = std::move(inputs);
    se->doOnlyApply();
    return std::move(se->outputs);
}

ZENO_API std::map<std::string, zany> Graph::callTempNode(std::string const &id,
        std::map<std::string, zany> inputs) const {
    auto cl = safe_at(session->nodeClasses, id, "node class name").get();
    auto se = cl->new_instance();
    se->graph = const_cast<Graph *>(this);
    se->inputs = std::move(inputs);
    se->doOnlyApply();
    return std::move(se->outputs);
}

ZENO_API void Graph::setObjRunType(std::string const& id, std::string const& type)
{
    safe_at(nodes, id, "node name")->objRunType = type;
}

ZENO_API INode* Graph::getNode(std::string const& id)
{
    return safe_at(nodes, id, "node name").get();
}

ZENO_API void Graph::addNodeOutput(std::string const& id, std::string const& par) {
    // add "dynamic" output which is not descriped by core.
    safe_at(nodes, id, "node name")->outputs[par] = nullptr;
}

ZENO_API void Graph::setNodeParam(std::string const &id, std::string const &par,
    std::variant<int, float, std::string, zany> const &val) {
    auto parid = par + ":";
    std::visit([&] (auto const &val) {
        using T = std::decay_t<decltype(val)>;
        if constexpr (std::is_same_v<T, zany>) {
            setNodeInput(id, parid, val);
        } else {
            setNodeInput(id, parid, objectFromLiterial(val));
        }
    }, val);
}

ZENO_API DirtyChecker &Graph::getDirtyChecker() {
    if (!dirtyChecker)
        dirtyChecker = std::make_unique<DirtyChecker>();
    return *dirtyChecker;
}

}
