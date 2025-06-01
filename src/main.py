import os
import argparse
from tqdm import tqdm
from . import MappedLanguageModel, TwoTimePadDecoder, parse_emails, evaluate_recovery


def main():
    parser = argparse.ArgumentParser(description="Two-time Pad Cracker")
    parser.add_argument("--corpus", required=True, help="Path to training corpus")
    parser.add_argument("--encrypted_dir", required=True, help="Directory with encrypted pairs")
    parser.add_argument("--model_path", default="email_model.bin", help="Model output path")
    parser.add_argument("--beam_width", type=int, default=100, help="Beam search width")
    args = parser.parse_args()

    # Load or train model
    model = MappedLanguageModel.load_or_train(
        args.corpus,
        args.model_path
    )

    # Load encrypted pairs
    encrypted_pairs = []
    for fname in os.listdir(args.encrypted_dir):
        if fname.endswith(".pair"):
            path = os.path.join(args.encrypted_dir, fname)
            with open(path, 'rb') as f:
                email1 = f.read()
                email2 = f.read()
                encrypted_pairs.append((email1, email2))

    # Initialize decoder
    decoder = TwoTimePadDecoder(model, model, beam_width=args.beam_width)

    # Process all pairs
    results = []
    for email1, email2 in tqdm(encrypted_pairs, desc="Decrypting"):
        xor_stream = bytes(a ^ b for a, b in zip(email1, email2))
        p1, p2 = decoder.decode(xor_stream)
        results.append((p1, p2))

    # Save results
    for i, (p1, p2) in enumerate(results):
        with open(f"decrypted_{i}_1.txt", 'wb') as f:
            f.write(p1)
        with open(f"decrypted_{i}_2.txt", 'wb') as f:
            f.write(p2)


if __name__ == "__main__":
    main()