# Two-Time Pad Cracker

This project implements the method described in the paper "A Natural Language Approach to Automated Cryptanalysis of Two-time Pads" by Mason et al. The goal is to recover plaintexts from two-time pad encrypted messages using statistical language models and dynamic programming. This implementation is in C++ and includes several optimizations and additional features.

## Main Components

1. **Language Models**: Implements n-gram character language models to estimate the probability of character sequences in plaintexts.
2. **Viterbi Search**: Uses dynamic programming with beam search to find the most probable pair of plaintexts given the XOR of their ciphertexts.
3. **Decryptor**: Orchestrates the decryption process, combining language models and Viterbi search.
4. **Visualizers**: Provides progress visualization and Viterbi graph visualization.
5. **Model Analyzer**: Analyzes language models, showing top n-grams and character distributions.
6. **GPU Accelerator**: Implements GPU acceleration for faster decryption (optional).

## Project Structure

```
project_root/
├── include/
│   ├── config_manager.h
│   ├── decryptor.h
│   ├── gpu_accelerator.h
│   ├── language_model.h
│   ├── logger.h
│   ├── model_analyzer.h
│   ├── visualizer.h
│   ├── viterbi_search.h
│   └── viterbi_visualizer.h
├── src/
│   ├── config_manager.cpp
│   ├── decryptor.cpp
│   ├── gpu_accelerator.cpp
│   ├── language_model.cpp
│   ├── logger.cpp
│   ├── main.cpp
│   ├── model_analyzer.cpp
│   ├── visualizer.cpp
│   ├── viterbi_search.cpp
│   └── viterbi_visualizer.cpp
├── test/
│   ├── include/
│   │   ├── test_config_manager.h
│   │   ├── test_decryptor.h
│   │   ├── test_gpu_accelerator.h
│   │   ├── test_language_model.h
│   │   ├── test_visualizer.h
│   │   └── test_viterbi_search.h
│   ├── src/
│   │   ├── test_config_manager.cpp
│   │   ├── test_decryptor.cpp
│   │   ├── test_gpu_accelerator.cpp
│   │   ├── test_language_model.cpp
│   │   ├── test_visualizer.cpp
│   │   └── test_viterbi_search.cpp
│   └── main_test.cpp
├── config/
│   └── config.yml
├── kernels/
│   └── viterbi_kernel.cl
├── CMakeLists.txt
└── README.md
```

## Key Features

- N-gram language models with Witten-Bell smoothing
- Viterbi algorithm with beam search for efficient decryption
- Support for multiple plaintext types (HTML, email, Word documents)
- GPU acceleration using OpenCL (optional)
- Visualization of decryption progress and Viterbi graph
- Configurable settings via YAML configuration file
- Logging system for better debugging and analysis

## Build and Run

1. Ensure you have a C++17 compatible compiler, CMake (3.15+), and the required libraries installed.
2. Clone the repository and navigate to the project root.
3. Create and navigate to a build directory:
   ```
   mkdir build && cd build
   ```
4. Configure the project with CMake:
   ```
   cmake ..
   ```
5. Build the project:
   ```
   cmake --build .
   ```
6. Run the program:
   ```
   ./bin/two_time_pad_cracker -c ../config/config.yml -x path/to/xored_text.txt
   ```

## Dependencies

- C++17 compatible compiler
- CMake 3.15+
- Boost (for graph visualization)
- yaml-cpp (for configuration management)
- cxxopts (for command-line argument parsing)
- spdlog (for logging)
- OpenCL (for GPU acceleration, optional)
- GTest (for unit testing)

## Configuration

The program uses a YAML configuration file for various settings. Example:

```yaml
model_file1: "path/to/model1.bin"
model_file2: "path/to/model2.bin"
num_threads: 4
pruning_threshold: 1e-5
verbose_mode: true
use_gpu: false
```

## Usage

```
two_time_pad_cracker --config path/to/config.yml --xor path/to/xored_text.txt
```

Use `--help` for more information on available options.

## Testing

The project includes unit tests using the Google Test framework. To run the tests:

1. Make sure you've built the project with the `BUILD_TESTS` option enabled:
   ```
   cmake -DBUILD_TESTS=ON ..
   cmake --build .
   ```
2. Run the tests:
   ```
   ctest
   ```
   or run the test executable directly:
   ```
   ./bin/unit_tests
   ```

## Performance

The implementation includes several optimizations:
- Beam search in Viterbi algorithm for memory efficiency
- Multi-threading support for parallel processing
- Optional GPU acceleration using OpenCL

On a typical modern PC, the program can process ciphertexts at approximately 200ms per byte.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.