import torch
import argparse
from torch.amp import autocast
from data.utils import TikTokenizer
from models.TransformerLM import TransformerLM


config = {
    "max_len": 2048,
}

def load_checkpoint(path, model, device):
    ckpt = torch.load(path, map_location=device)
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    raw_model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint from {path} (step {ckpt.get('step', '?')})")


def tokenize_conversation(turns, tokenizer):
    text = []
    for role, content in turns:
        text += tokenizer.encode(f"<|{role}|> {content} ") + [tokenizer.pad_id]
    text += tokenizer.encode("<|assistant|>")
    return text


def generate_response(model, tokenizer, device, turns, max_new_tokens=500):
    tokens = tokenize_conversation(turns, tokenizer)
    
    # Truncate from the left if the prompt is too long, keeping room for the response
    max_prompt = config["max_len"] - max_new_tokens
    if len(tokens) > max_prompt:
        tokens = tokens[-max_prompt:]

    x = torch.tensor([tokens], dtype=torch.long).to(device)

    with torch.no_grad():
        with autocast(device_type=device, dtype=torch.float16):
            out = model.generate(x, max_new_tokens=max_new_tokens, temp=0.7, top_p=0.95)[0].tolist()

    # Decode only the newly generated tokens
    response_tokens = out[len(tokens):]
    response = tokenizer.decode(response_tokens)

    # Trim at special tokens
    if "<|assistant|>" in response:
        response = response[:response.index("<|assistant|>")]

    if "<|user|>" in response:
        response = response[:response.index("<|user|>")]
    
    if "<|endoftext|>" in response:
        response = response[:response.index("<|endoftext|>")]

    return response.strip()


def main():
    parser = argparse.ArgumentParser(description="Chat with a fine-tuned TransformerLM checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to fine-tuned checkpoint, e.g. chkpts/finetune_epoch3.pt")
    parser.add_argument("--max_new_tokens", type=int, default=200,
                        help="Maximum tokens to generate per response")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = TikTokenizer()
    model = TransformerLM(tokenizer.vocab_size, max_len=config["max_len"]).to(device)
    load_checkpoint(args.checkpoint, model, device)
    model.eval()


    print("\nChat session started. Type 'quit' or 'exit' to end, 'reset' to start a new conversation.\n")

    turns = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Exiting.")
            break
        if user_input.lower() == "reset":
            turns = []
            print("--- Conversation reset ---\n\n\n\n")
            continue

        turns.append(("user", user_input))
        response = generate_response(model, tokenizer, device, turns, args.max_new_tokens)
        turns.append(("assistant", response))

        print(f"\nAssistant: {response} \n")

        

if __name__ == "__main__":
    main()