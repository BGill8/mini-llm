import argparse
import torch
from torch.amp import autocast
from data.utils import TikTokenizer
from models.TransformerLM import TransformerLM
 
 
def load_checkpoint(path, device):
    print(f"Loading checkpoint from: {path}")
    ckpt = torch.load(path, map_location=device)
 
    config = ckpt["config"]
    max_len = config.get("max_len", 1024)
 
    tokenizer = TikTokenizer()
    model = TransformerLM(tokenizer.vocab_size, max_len=max_len).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
 
    step = ckpt.get("step", "?")
    print(f"  Resumed from training step: {step}")
    print(f"  Model max_len: {max_len}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
 
    return model, tokenizer
 
 
def generate(model, tokenizer, prompt, device, max_new_tokens=200, use_amp=True):
    ids = tokenizer.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
 
    with torch.no_grad():
        if use_amp and device == "cuda":
            with autocast(device_type="cuda", dtype=torch.float16):
                out_ids = model.generate(x, max_new_tokens=max_new_tokens)[0].tolist()
        else:
            out_ids = model.generate(x, max_new_tokens=max_new_tokens)[0].tolist()
 
    return tokenizer.decode(out_ids)
 
 
def main():
    parser = argparse.ArgumentParser(description="Query a TransformerLM checkpoint interactively.")
    parser.add_argument("--checkpoint", help="Path to the .pt checkpoint file (e.g. chkpts/pretrain_latest.pt)")
    parser.add_argument("--max-new-tokens", type=int, default=200, help="Max tokens to generate per reply (default: 200)")
    parser.add_argument("--no-amp", action="store_true", help="Disable FP16 autocast even on CUDA")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    args = parser.parse_args()
 
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    print(f"Using device: {device}\n")
 
    model, tokenizer = load_checkpoint(args.checkpoint, device)
    use_amp = not args.no_amp
 
    print("\nModel ready. Type your prompt and press Enter. Type 'quit' or Ctrl-C to exit.\n")
    print("=" * 60)
 
    while True:
        try:
            prompt = input(">>> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
 
        if not prompt:
            continue
        if prompt.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
 
        reply = generate(model, tokenizer, prompt, device,
                         max_new_tokens=args.max_new_tokens,
                         use_amp=use_amp)
        print("\n" + reply + "\n" + "=" * 60)
 
 
if __name__ == "__main__":
    main()