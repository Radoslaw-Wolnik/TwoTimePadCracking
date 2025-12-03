#!/usr/bin/env python3
"""
Main entry point for the Two-Time Pad Decryption System
"""

import argparse
import logging
import sys
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from src.data.loader import DataManager
from src.data.enron import EnronDownloader, EnronPreprocessor
from src.data.html_corpus import HTMLCorpusDownloader
from src.model.char_language_model import CharLanguageModel
from src.model.decoder import TwoTimePadDecoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TwoTimePadCLI:
    def __init__(self, res_dir: Path = Path("res")):
        self.res_dir = Path(res_dir)
        self.data_manager = DataManager(self.res_dir)
        
    def setup_data(self, data_type: str):
        """Download and prepare data for training"""
        logger.info(f"Setting up data for: {data_type}")
        
        if data_type == "email":
            self._setup_enron()
        elif data_type == "html":
            self._setup_html()
        elif data_type == "books":
            self._setup_books()
        else:
            raise ValueError(f"Unknown data type: {data_type}")
            
        logger.info(f"Data setup complete for {data_type}")
    
    def _setup_enron(self):
        """Setup Enron email data"""
        downloader = EnronDownloader(self.data_manager.raw_dir)
        tgz_path = downloader.download()
        
        # Extract to raw directory
        extract_dir = self.data_manager.raw_dir / "enron_extracted"
        downloader.extract(tgz_path, extract_dir)
        
        # Preprocess emails
        preprocessor = EnronPreprocessor(extract_dir, self.data_manager.processed_dir)
        count = preprocessor.process()
        logger.info(f"Processed {count} emails")
    
    def _setup_html(self):
        """Setup HTML corpus data"""
        # This would need a list of URLs - for now we'll create a basic implementation
        downloader = HTMLCorpusDownloader(self.data_manager.processed_dir)
        
        # Example URLs - in practice you'd want a larger list
        urls = [
            "https://example.com/page1",
            "https://example.com/page2",
        ]
        
        success_count = 0
        for i, url in enumerate(urls):
            if downloader.save_html_as_text(url, i):
                success_count += 1
                
        logger.info(f"Successfully downloaded {success_count} HTML pages")
    
    def _setup_books(self):
        """Setup book corpus data"""
        # Placeholder for Gutenberg integration
        logger.info("Book data setup - Gutenberg integration not yet implemented")
        # Would integrate with gutenberg.py here
    
    def train_model(self, data_type: str, model_name: str, open_existing: bool = False):
        """Train a language model on the specified data type"""
        logger.info(f"Training model for {data_type} with name {model_name}")
        
        model_path = self.data_manager.models_dir / f"{model_name}.bin"
        
        if open_existing and not model_path.exists():
            raise FileNotFoundError(f"Model {model_name} not found at {model_path}")
        
        if open_existing:
            logger.info(f"Loading existing model: {model_name}")
            model = CharLanguageModel.load(model_path)
        else:
            logger.info(f"Creating new model: {model_name}")
            model = CharLanguageModel(n=7)
        
        # Load training data
        texts = list(self.data_manager.iter_texts(self.data_manager.processed_dir))
        if not texts:
            raise ValueError(f"No training data found in {self.data_manager.processed_dir}")
        
        logger.info(f"Training on {len(texts)} documents")
        
        # Train the model
        for i, text in enumerate(texts):
            if i % 100 == 0:
                logger.info(f"Processing document {i}/{len(texts)}")
            model.train(text)
        
        # Save the model
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")
    
    def decode(self, model_name: str, doc_type: str, file1_path: str, file2_path: str):
        """Decode two encrypted texts using the specified model"""
        logger.info(f"Decoding with model {model_name} for {doc_type}")
        
        # Load model
        model_path = self.data_manager.models_dir / f"{model_name}.bin"
        if not model_path.exists():
            raise FileNotFoundError(f"Model {model_name} not found at {model_path}")
        
        model = CharLanguageModel.load(model_path)
        
        # For now, we use the same model for both texts
        # In a more advanced setup, you might have different models for different doc types
        decoder = TwoTimePadDecoder(model1=model, model2=model, beam_width=100)
        
        # Read the encrypted files
        with open(file1_path, 'rb') as f1, open(file2_path, 'rb') as f2:
            encrypted1 = f1.read()
            encrypted2 = f2.read()
        
        # XOR the encrypted texts to get the XOR stream
        if len(encrypted1) != len(encrypted2):
            logger.warning("Encrypted files have different lengths, using minimum length")
            min_len = min(len(encrypted1), len(encrypted2))
            encrypted1 = encrypted1[:min_len]
            encrypted2 = encrypted2[:min_len]
        
        xor_stream = [a ^ b for a, b in zip(encrypted1, encrypted2)]
        
        # Decode
        logger.info("Starting decoding...")
        plaintext1, plaintext2 = decoder.decode(xor_stream)
        
        # Save results
        output_dir = self.res_dir / "decoded"
        output_dir.mkdir(exist_ok=True)
        
        output1 = output_dir / f"{Path(file1_path).stem}_decoded.txt"
        output2 = output_dir / f"{Path(file2_path).stem}_decoded.txt"
        
        with open(output1, 'wb') as f:
            f.write(plaintext1)
        with open(output2, 'wb') as f:
            f.write(plaintext2)
        
        logger.info(f"Decoded texts saved to:")
        logger.info(f"  {output1}")
        logger.info(f"  {output2}")
        
        return plaintext1, plaintext2


def main():
    parser = argparse.ArgumentParser(
        description="Two-Time Pad Decryption System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Setup data for training
  python main.py --setup --type email
  
  # Train a new model
  python main.py --train --type email --model-name email_model --new
  
  # Continue training existing model
  python main.py --train --type email --model-name email_model --open-model-name email_model
  
  # Decode encrypted files
  python main.py --decoding --model-name email_model --doc-type email --file1path enc1.bin --file2path enc2.bin
        """
    )
    
    # Main action flags (mutually exclusive)
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument('--setup', action='store_true', help='Setup training data')
    action_group.add_argument('--train', action='store_true', help='Train a model')
    action_group.add_argument('--decoding', action='store_true', help='Decode encrypted texts')
    
    # Common arguments
    parser.add_argument('--type', '--doc-type', choices=['email', 'html', 'books'], 
                       help='Type of data/document')
    parser.add_argument('--model-name', help='Name of the model (for training/decoding)')
    parser.add_argument('--res-dir', default='res', help='Resources directory (default: res)')
    
    # Training-specific arguments
    parser.add_argument('--new', action='store_true', 
                       help='Create new model (for training)')
    parser.add_argument('--open-model-name', 
                       help='Open existing model for continued training')
    
    # Decoding-specific arguments  
    parser.add_argument('--file1path', help='Path to first encrypted file')
    parser.add_argument('--file2path', help='Path to second encrypted file')
    
    args = parser.parse_args()
    
    # Validate arguments based on action
    if args.setup and not args.type:
        parser.error("--setup requires --type")
    
    if args.train:
        if not args.type or not args.model_name:
            parser.error("--train requires --type and --model-name")
        if not args.new and not args.open_model_name:
            parser.error("--train requires either --new or --open-model-name")
    
    if args.decoding:
        if not args.model_name or not args.type or not args.file1path or not args.file2path:
            parser.error("--decoding requires --model-name, --doc-type, --file1path, and --file2path")
    
    # Initialize CLI
    cli = TwoTimePadCLI(Path(args.res_dir))
    
    try:
        if args.setup:
            cli.setup_data(args.type)
            
        elif args.train:
            open_existing = bool(args.open_model_name)
            model_to_open = args.open_model_name or args.model_name
            cli.train_model(args.type, model_to_open, open_existing)
            
        elif args.decoding:
            cli.decode(args.model_name, args.type, args.file1path, args.file2path)
            
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()