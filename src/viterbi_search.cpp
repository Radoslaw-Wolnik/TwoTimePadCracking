// viterbi_search.cpp
#include "viterbi_search.h"
#include "logger.h"
#include <algorithm>
#include <limits>
#include <iostream>
#include <sstream>


ViterbiSearch::ViterbiSearch(const LanguageModel& model1, const LanguageModel& model2)
    : model1_(model1), model2_(model2) {}

std::pair<std::string, std::string> ViterbiSearch::decrypt(
    const std::string& xor_text,
    std::function<void(size_t, double)> progress_callback,
    std::function<void(int, const std::string&, double)> visualizer_callback
) {
    visualizer_callback_ = visualizer_callback;
    std::vector<std::shared_ptr<State>> current_states = {std::make_shared<State>("", "", 0.0)};
    
    for (size_t i = 0; i < xor_text.length(); ++i) {
        if (cancel_requested_) {
            throw std::runtime_error("Decryption cancelled");
        }

        current_states = viterbi_step(current_states, xor_text[i]);
        prune_states(current_states);
        
        // Calculate and report progress
        auto best_state = *std::max_element(current_states.begin(), current_states.end(),
            [](const auto& a, const auto& b) { return a->probability < b->probability; });
        double confidence = std::exp(best_state->probability / (i + 1));  // Normalize by length
        progress_callback(i + 1, confidence);

        log_verbose("Step " + std::to_string(i + 1) + ": " + std::to_string(current_states.size()) + " states, best prob: " + std::to_string(best_state->probability));
    }
    
    auto best_state = *std::max_element(current_states.begin(), current_states.end(),
        [](const auto& a, const auto& b) { return a->probability < b->probability; });
    
    return backtrack(best_state, xor_text.length());
}

void ViterbiSearch::visualize_state(const State& state, int step) {
    std::string state_id = state.context1 + "|" + state.context2;
    visualizer_callback_(step, state_id, state.probability);
}

std::vector<std::shared_ptr<ViterbiSearch::State>> ViterbiSearch::viterbi_step(
    const std::vector<std::shared_ptr<State>>& prev_states, char xor_char) {
    std::vector<std::shared_ptr<State>> new_states;
    
    for (const auto& prev_state : prev_states) {
        for (int char1 = 0; char1 < 256; ++char1) {
            char char2 = char1 ^ xor_char;
            
            double prob1 = model1_.getProbability(prev_state->context1, char1);
            double prob2 = model2_.getProbability(prev_state->context2, char2);
            double new_prob = prev_state->probability + std::log(prob1) + std::log(prob2);
            
            auto new_state = std::make_shared<State>(
                prev_state->context1 + static_cast<char>(char1),
                prev_state->context2 + static_cast<char>(char2),
                new_prob
            );
            new_state->prev_state = prev_state;
            new_state->char1 = char1;
            new_state->char2 = char2;
            
            new_states.push_back(new_state);
        }
    }
    
    return new_states;
}

std::pair<std::string, std::string> ViterbiSearch::backtrack(std::shared_ptr<State> final_state, size_t length) {
    std::string text1(length, ' '), text2(length, ' ');
    auto current_state = final_state;
    
    for (int i = length - 1; i >= 0; --i) {
        text1[i] = current_state->char1;
        text2[i] = current_state->char2;
        current_state = current_state->prev_state;
    }
    
    return {text1, text2};
}

void ViterbiSearch::prune_states(std::vector<std::shared_ptr<State>>& states) {
    if (states.empty()) return;

    // Sort states by probability (descending order)
    std::sort(states.begin(), states.end(),
        [](const auto& a, const auto& b) { return a->probability > b->probability; });

    // Find the pruning threshold
    double best_prob = states[0]->probability;
    double threshold = best_prob + std::log(pruning_threshold_);

    // Find the last state above the threshold
    auto it = std::find_if(states.begin(), states.end(),
        [threshold](const auto& state) { return state->probability < threshold; });

    // Prune states
    states.erase(it, states.end());

    log_verbose("Pruned states from " + std::to_string(states.size()) + " to " + std::to_string(std::distance(states.begin(), it)));
}

void ViterbiSearch::log_verbose(const std::string& message) const {
    if (verbose_) {
        std::cout << "[ViterbiSearch] " << message << std::endl;
    }
}



std::vector<std::shared_ptr<ViterbiSearch::State>> ViterbiSearch::viterbi_step_parallel(
    const std::vector<std::shared_ptr<State>>& prev_states, char xor_char) {
    std::vector<std::shared_ptr<State>> new_states;
    std::mutex new_states_mutex;

    auto worker = [&](size_t start, size_t end) {
        std::vector<std::shared_ptr<State>> local_new_states;
        for (size_t i = start; i < end; ++i) {
            const auto& prev_state = prev_states[i];
            for (int char1 = 0; char1 < 256; ++char1) {
                char char2 = char1 ^ xor_char;
                
                double prob1 = model1_.getProbability(prev_state->context1, char1);
                double prob2 = model2_.getProbability(prev_state->context2, char2);
                double new_prob = prev_state->probability + std::log(prob1) + std::log(prob2);
                
                auto new_state = std::make_shared<State>(
                    prev_state->context1 + static_cast<char>(char1),
                    prev_state->context2 + static_cast<char>(char2),
                    new_prob
                );
                new_state->prev_state = prev_state;
                new_state->char1 = char1;
                new_state->char2 = char2;
                
                local_new_states.push_back(new_state);
            }
        }
        std::lock_guard<std::mutex> lock(new_states_mutex);
        new_states.insert(new_states.end(), local_new_states.begin(), local_new_states.end());
    };

    std::vector<std::thread> threads;
    size_t chunk_size = prev_states.size() / num_threads_;
    for (int i = 0; i < num_threads_; ++i) {
        size_t start = i * chunk_size;
        size_t end = (i == num_threads_ - 1) ? prev_states.size() : (i + 1) * chunk_size;
        threads.emplace_back(worker, start, end);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    return new_states;
}

std::vector<std::shared_ptr<ViterbiSearch::State>> ViterbiSearch::viterbi_step_gpu(
    const std::vector<std::shared_ptr<State>>& prev_states, char xor_char) {
    // This is a placeholder for GPU acceleration
    // You'll need to implement this using your GPU accelerator
    Logger::get()->warn("GPU acceleration not yet implemented, falling back to CPU");
    return viterbi_step_parallel(prev_states, xor_char);
}

std::vector<std::shared_ptr<ViterbiSearch::State>> ViterbiSearch::viterbi_step(
    const std::vector<std::shared_ptr<State>>& prev_states, char xor_char) {
    if (use_gpu_) {
        return viterbi_step_gpu(prev_states, xor_char);
    } else if (num_threads_ > 1) {
        return viterbi_step_parallel(prev_states, xor_char);
    } else {
        // Existing single-threaded implementation
        // ... (keep existing implementation)
    }
}

void ViterbiSearch::visualize_state(const State& state, int step) {
    if (visualizer_) {
        std::string state_id = state.context1 + "|" + state.context2;
        visualizer_->addState(step, state_id, state.probability);
    }
}

void ViterbiSearch::viterbi_step_worker(const std::vector<std::shared_ptr<State>>& prev_states, 
                                        char xor_char, 
                                        size_t start, 
                                        size_t end, 
                                        std::vector<std::shared_ptr<State>>& new_states,
                                        std::mutex& new_states_mutex) {
    std::vector<std::shared_ptr<State>> local_new_states;
    
    for (size_t i = start; i < end; ++i) {
        const auto& prev_state = prev_states[i];
        for (int char1 = 0; char1 < 256; ++char1) {
            char char2 = char1 ^ xor_char;
            
            double prob1 = model1_.getProbability(prev_state->context1, char1);
            double prob2 = model2_.getProbability(prev_state->context2, char2);
            double new_prob = prev_state->probability + std::log(prob1) + std::log(prob2);
            
            auto new_state = std::make_shared<State>(
                prev_state->context1 + static_cast<char>(char1),
                prev_state->context2 + static_cast<char>(char2),
                new_prob
            );
            new_state->prev_state = prev_state;
            new_state->char1 = char1;
            new_state->char2 = char2;
            
            local_new_states.push_back(new_state);
        }
    }

    std::lock_guard<std::mutex> lock(new_states_mutex);
    new_states.insert(new_states.end(), local_new_states.begin(), local_new_states.end());
}
