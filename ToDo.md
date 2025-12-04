## Future Work & Improvements

### Short-Term
1. **Implement multiple keystream reuse** (3+ texts) to reduce switching
2. **Add post-processing switch detection** as described in the paper
3. **Support for mixed document types** (HTML + email decoding)
4. **Adaptive beam width** based on decoding confidence

### Medium-Term
2. **GPU acceleration** for faster beam search
3. **Real-time decoding** for live traffic analysis

### Long-Term
1. **Universal decoder** for arbitrary document types
2. **Cryptanalysis as a Service** API
3. **Educational platform** with interactive visualizations
4. **Integration with network analyzers** for protocol-level attacks (this makes sense as some web packages still use same key decryption) - this is nice to do in future

### Actually what i want to do
1. Make c# or C++ implementation of [CharLanguageModel](./src/model/char_language_model.py) [Decoder](./src/model/decoder.py) and [MappedLanguageModel](./src/model/mapped_language_model.py)
2. Make usage of Mapped language model
3. Make it possible to download and preprocess book data nad html data and word data based documents


<!--
## Running 

`python -m src.main --setup --type email`

`python -m src.main --train --type email --model-name email_model --new `

`python -m src.main --train --type email --model-name email_model --open-model-name email_model`

`python -m src.main --decoding --model-name email_model --doc-type email --file1path enc1.bin --file2path enc2.bin`

### From the project root directory
python -m pytest src/tests/ -v

### Run specific test files
python -m pytest src/tests/unit/test_char_language_model.py -v
python -m pytest src/tests/integration/test_decoder_integration.py -v

### Run with markers (like slow tests)
python -m pytest src/tests/ -m slow -v
-->


