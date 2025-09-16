import numpy as np

import torch.nn.functional as F
import torch

def cosine_similarity(a: np.ndarray, b: np.ndarray, shape: tuple) -> float:
    a, b = torch.tensor(a), torch.tensor(b)

    print("DEBUG: a: ", a.shape)
    print("DEBUG: b: ", b.shape)

    a = a.reshape(shape)
    b = b.reshape(shape)

    return F.cosine_similarity(a, b, dim=-1), (a - b)

import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Compare Ollama and Transformers model outputs")
    parser.add_argument("ollama_output", help="Path to Ollama output binary file")
    parser.add_argument("transformers_output", help="Path to Transformers output binary file")
    parser.add_argument("--size", nargs="+", type=int, required=True, 
                       help="Shape dimensions (e.g., --size 4 7168)")
    
    args = parser.parse_args()

    x = np.fromfile(args.ollama_output, dtype=np.float32)
    y = np.fromfile(args.transformers_output, dtype=np.float32)

    # x = np.fromfile(args.ollama_output, dtype=np.int32)
    # y = np.fromfile(args.transformers_output, dtype=np.int32)

    # x.astype(np.float32)
    # y.astype(np.float32)

    # # x.astype(int)
    # # y.astype(int)

    # print("DEBUG: x: ", x)
    # print("DEBUG: y: ", y)
    
    try:
        size = tuple(args.size)
        cos_sim, diff = cosine_similarity(x, y, size)

        print(f"Cosine Similarity - Mean: {cos_sim.mean():.6f}, Max: {cos_sim.max():.6f}")
        print(f"Difference - Mean: {diff.mean():.6f}, Max: {diff.max():.6f}")
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

