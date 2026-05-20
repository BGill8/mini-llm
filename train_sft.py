import torch
import argparse
from data.utils import TikTokenizer
from models.TransformerLM import TransformerLM
from data.SmolTalk import getSmolTalkDataloader
import wandb
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import math
import os


config = {
    "bs": 7,
    "max_len": 1024,
    "lr": 1e-4,
    "warmup_steps": 200,
    "epochs": 3,
}


def load_checkpoint(path, model, device):
    print(f"Loading checkpoint from {path}")
    ckpt = torch.load(path, map_location=device)
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    raw_model.load_state_dict(ckpt["model"])
    print(f"Loaded pretrained weights from step {ckpt.get('step', '?')}")


def masked_cross_entropy(logits, targets, mask):
    loss_per_token = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none"
    )
    mask_flat = mask.reshape(-1).float()
    return (loss_per_token * mask_flat).sum() / mask_flat.sum().clamp(min=1)


def finetune(model, train_loader, test_loader, tokenizer, device, checkpoint_path):

    wandb.login()
    wandb.init(project="[AI539] SmolTalk Finetune", config=config)

    print("Total Parameters: " + str(sum(p.numel() for p in model.parameters())))

    grad_accum_steps = math.ceil(32768 // (config["bs"] * config["max_len"]))
    print("Batches per update: " + str(grad_accum_steps))
    config["grad_accum_steps"] = grad_accum_steps

    # Total updates across all epochs — used only for the cosine schedule
    steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
    total_steps = steps_per_epoch * config["epochs"]
    print(f"Steps per epoch: {steps_per_epoch} | Total steps: {total_steps}")

    opt = torch.optim.AdamW(model.parameters(), lr=config["lr"], betas=(0.9, 0.95), weight_decay=0.1)
    scaler = GradScaler()

    linear = LinearLR(opt, start_factor=0.05, total_iters=config["warmup_steps"])
    cosine = CosineAnnealingLR(opt, T_max=total_steps - config["warmup_steps"], eta_min=config["lr"] / 10)
    scheduler = SequentialLR(opt, schedulers=[linear, cosine], milestones=[config["warmup_steps"]])

    # Load pretrained weights only — fresh optimizer for fine-tuning
    load_checkpoint(checkpoint_path, model, device)

    model.train()
    global_step = 0
    loss_tracker = 0

    for epoch in range(config["epochs"]):
        
        pbar = tqdm(total=steps_per_epoch, desc=f"Epoch {epoch+1}", unit="updates")

        for i, (tokens, mask) in enumerate(train_loader):
            
            tokens = tokens.to(device)
            mask = mask.to(device)

            with autocast(device_type="cuda", dtype=torch.float16):
                logits = model(tokens[:, :-1])
                loss = masked_cross_entropy(
                    logits,
                    tokens[:, 1:],
                    mask[:, 1:]
                ) / grad_accum_steps

            scaler.scale(loss).backward()
            loss_tracker += loss.item()

            if (i+1) % grad_accum_steps == 0:

                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(opt)
                scaler.update()
                scheduler.step()
                
                opt.zero_grad()
                
                

                wandb.log({
                    "Loss/train": loss_tracker,
                    "Misc/lr": scheduler.get_last_lr()[0],
                    "Misc/epoch": epoch + 1,
                }, step=global_step)


                loss_tracker = 0
                global_step += 1
                pbar.update(1)

                # Checkpointing
                if (global_step + 1) % 250 == 0:
                    val_loss = evaluate(model, test_loader, device, num_batches=5)
                    wandb.log({"Loss/val": val_loss}, step=global_step)
                    _save_checkpoint(model, opt, scaler, scheduler, global_step, epoch)

                # Sample generation
                if (global_step + 1) % 250 == 0:
                    _generate_sample(model, tokenizer, device, global_step)
                    model.train()

        pbar.close()

        # Checkpoint at end of each epoch
        _save_checkpoint(model, opt, scaler, scheduler, global_step, epoch)


def evaluate(model, test_loader, device, num_batches=5):
    """Compute mean masked cross-entropy loss over `num_batches` validation batches."""
    model.eval()
    total_loss = 0.0
 
    with torch.no_grad():
        for i, (tokens, mask) in enumerate(test_loader):
            if i >= num_batches:
                break
 
            tokens = tokens.to(device)
            mask = mask.to(device)
 
            with autocast(device_type="cuda", dtype=torch.float16):
                logits = model(tokens[:, :-1])
                loss = masked_cross_entropy(logits, tokens[:, 1:], mask[:, 1:])
 
            total_loss += loss.item()
 
    model.train()
    return total_loss / num_batches

def _save_checkpoint(model, opt, scaler, scheduler, step, epoch):
    os.makedirs("chkpts", exist_ok=True)
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save({
        "model": raw_model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "epoch": epoch,
        "config": config,
    }, f"chkpts/sft_epoch{epoch+1}.pt")


def _generate_sample(model, tokenizer, device, step):
    model.eval()
    with torch.no_grad():
        prompt = "<|user|>\nExplain transformers simply.\n<|assistant|>\n"
        x = torch.tensor([tokenizer.encode(prompt)]).to(device)
        with autocast(device_type="cuda", dtype=torch.float16):
            out = model.generate(x)[0].tolist()
        sample = tokenizer.decode(out)
        wandb.log({"sample_text": wandb.Html(f"<pre>{sample}</pre>")}, step=step)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune on SmolTalk")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to pretrained checkpoint, e.g. chkpts/pretrain_latest.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = TikTokenizer()
    model = TransformerLM(tokenizer.vocab_size, max_len=config["max_len"]).to(device)
    model = torch.compile(model, dynamic=False)
    train_loader, test_loader = getSmolTalkDataloader(tokenizer, batch_size=config["bs"], max_length=config["max_len"])

    print("=== FINETUNE ===")
    finetune(model, train_loader, test_loader, tokenizer, device, args.checkpoint)


if __name__ == "__main__":
    main()