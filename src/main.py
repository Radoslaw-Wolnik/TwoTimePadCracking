# run_cracker.py
import argparse
import os
from src import CharLanguageModel, TwoTimePadDecoder, evaluate_recovery


def main():
    parser = argparse.ArgumentParser(description="Two-Time Pad Cracker")
    parser.add_argument("--model1", required=True, help="Path to first language model")
    parser.add_argument("--model2", required=True, help="Path to second language model")
    parser.add_argument("--xor_file", required=True, help="File containing XOR stream")
    parser.add_argument("--output_dir", required=True, help="Output directory for results")
    parser.add_argument("--beam_width", type=int, default=100)

    args = parser.parse_args()

    # Load models
    print("Loading language models...")
    model1 = CharLanguageModel.load(args.model1)
    model2 = CharLanguageModel.load(args.model2)

    # Load XOR stream
    with open(args.xor_file, 'rb') as f:
        xor_stream = f.read()

    # Decode
    print("Decoding...")
    decoder = TwoTimePadDecoder(model1, model2, beam_width=args.beam_width)
    plaintext1, plaintext2 = decoder.decode(xor_stream)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "decoded1.txt"), 'wb') as f:
        f.write(plaintext1)
    with open(os.path.join(args.output_dir, "decoded2.txt"), 'wb') as f:
        f.write(plaintext2)

    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()