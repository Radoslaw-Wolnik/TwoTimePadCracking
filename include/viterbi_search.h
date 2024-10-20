// viterbi_search.h
#pragma once

#include "language_model.h"
#include "viterbi_visualizer.h"
#include "gpu_accelerator.h"
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <atomic>
#include <mutex>
#include <thread>

class ViterbiSearch {
public:
    ViterbiSearch(const LanguageModel& model1, const LanguageModel& model2);
    std::pair<std::string, std::string> decrypt(
        const std::string& xor_text, 
        std::function<void(size_t, double)> progress_callback = [](size_t, double){},
        std::function<void(int, const std::string&, double)> visualizer_callback = [](int, const std::string&, double){}
    );

    void setPruningThreshold(double threshold) { pruning_threshold_ = threshold; }
    void setVerbose(bool verbose) { verbose_ = verbose; }
    void setNumThreads(int num_threads) { num_threads_ = num_threads; }
    void setUseGPU(bool use_gpu) { use_gpu_ = use_gpu; }
    void setVisualizer(std::shared_ptr<ViterbiVisualizer> visualizer) { visualizer_ = visualizer; }

private:
    const LanguageModel& model1_;
    const LanguageModel& model2_;
    double pruning_threshold_ = 1e-5;
    bool verbose_ = false;
    std::atomic<bool> cancel_requested_{false};
    int num_threads_ = std::thread::hardware_concurrency();
    bool use_gpu_ = false;
    std::shared_ptr<ViterbiVisualizer> visualizer_;
    std::unique_ptr<GPUAccelerator> gpu_accelerator_;
    
    struct State {
        std::string context1;
        std::string context2;
        double probability;
        std::shared_ptr<State> prev_state;
        char char1;
        char char2;
        
        State(const std::string& c1, const std::string& c2, double prob)
            : context1(c1), context2(c2), probability(prob), prev_state(nullptr), char1(0), char2(0) {}
    };
    
    std::vector<std::shared_ptr<State>> viterbi_step(const std::vector<std::shared_ptr<State>>& prev_states, char xor_char);
    std::vector<std::shared_ptr<State>> viterbi_step_parallel(const std::vector<std::shared_ptr<State>>& prev_states, char xor_char);
    std::vector<std::shared_ptr<State>> viterbi_step_gpu(const std::vector<std::shared_ptr<State>>& prev_states, char xor_char);
    std::pair<std::string, std::string> backtrack(std::shared_ptr<State> final_state, size_t length);
    void prune_states(std::vector<std::shared_ptr<State>>& states);
    void log_verbose(const std::string& message) const;
    void visualize_state(const State& state, int step);
};
