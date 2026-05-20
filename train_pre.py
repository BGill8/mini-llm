import torch
from data.utils import TikTokenizer
from models.TransformerLM import TransformerLM
from data.MiniPile import getMiniPileDataloder
import wandb
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from itertools import cycle
import math
import os
            

config = {
    "bs":7,   # batch size
    "max_len":1024, # max seqeunce length
    "lr":5e-4, # learning rate
    "warmup_steps":1000, 
}

def pretrain(model, loader, tokenizer, device, seq_len=1024):
    
    wandb.login()
    wandb.init(project="[AI539] MiniPile HW3")

    print("Total Parameters: " + str(sum(p.numel() for p in model.parameters())))


    # Calculate the gradient accumulation steps needed to get around 250k tokens per update
    grad_accum_steps = math.ceil(250000 // (config["bs"]*config["max_len"]))
    print("Batches per update: "+str(grad_accum_steps))
    config['grad_accum_steps'] = grad_accum_steps

    # Calculate how many update steps to make one epoch through the 1.7 billion tokens of training data
    max_steps = int(1.7e9 // (config["bs"]*config["max_len"]*grad_accum_steps))
    print("Updates for one epoch: "+str(max_steps))
    config['max_steps'] = max_steps

    
    muon_params = [p for n, p in model.named_parameters() if p.ndim >= 2 and 'embed' not in n]
    adamw_params = [p for n, p in model.named_parameters() if p.ndim < 2 or 'embed' in n]
    
    opt_adam = torch.optim.AdamW(adamw_params, lr=config['lr'], betas=(0.9, 0.95), weight_decay=0.1)
    opt_muon = torch.optim.Muon(muon_params, lr=0.02)
    
    scaler = GradScaler()
    
    linear_adam = LinearLR(opt_adam, start_factor=0.05, total_iters=config["warmup_steps"])
    cosine_adam = CosineAnnealingLR(opt_adam, T_max = max_steps - config["warmup_steps"],eta_min=config["lr"]/10)
    scheduler_adam = SequentialLR(opt_adam, schedulers=[linear_adam, cosine_adam], milestones=[config["warmup_steps"]])

    linear_muon = LinearLR(opt_muon, start_factor=0.05, total_iters=config["warmup_steps"])
    cosine_muon = CosineAnnealingLR(opt_muon, T_max = max_steps - config["warmup_steps"],eta_min=0.02/10)
    scheduler_muon = SequentialLR(opt_muon, schedulers=[linear_muon, cosine_muon], milestones=[config["warmup_steps"]])


    model.train()

    
    step = 0

    loader = cycle(loader)
    
    
    # Train loop
    pbar = tqdm(total=max_steps, desc="Training Steps", unit="updates")
    while step < max_steps:

        loss_tracker = 0
        opt_adam.zero_grad()
        opt_muon.zero_grad()
        for _ in range(grad_accum_steps):
            batch = next(loader).to(device)

            with autocast(device_type="cuda", dtype=torch.float16):
                logits = model(batch[:, :-1])
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    batch[:, 1:].reshape(-1)
                ) / grad_accum_steps

            scaler.scale(loss).backward()
            loss_tracker += loss.item()

        # Perform gradient clipping
        scaler.unscale_(opt_adam)
        scaler.unscale_(opt_muon)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Step model
        scaler.step(opt_adam)
        opt_muon.step()
        scaler.update()
        scheduler_adam.step()
        scheduler_muon.step()

        wandb.log({"Loss/train": loss_tracker, "Misc/lr":scheduler_adam.get_last_lr()[0]}, step=step)

        pbar.update(1)

        # ---- checkpointing ----
        if (step + 1) % 250 == 0:
            os.makedirs("chkpts", exist_ok=True)

            # handle torch.compile wrapper
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model

            torch.save({
                "model": raw_model.state_dict(),
                "optimizer": [opt_adam.state_dict(), opt_muon.state_dict()],
                "scheduler": [scheduler_adam.state_dict(), scheduler_muon.state_dict()],
                "scaler": scaler.state_dict(),
                "step": step,
                "config": config,
            }, "chkpts/pretrain_latest.pt")
        

        # sample generation
        if (step+1) % 500 == 0:
            model.eval()
            with torch.no_grad():
                prompt = "Once upon a time, there was a"
                x = torch.tensor([tokenizer.encode(prompt)]).to(device)

                with autocast(device_type="cuda", dtype=torch.float16):
                    out = model.generate(x)[0].tolist()

                sample = tokenizer.decode(out)

                wandb.log({
                    "sample_text": wandb.Html(f"<pre>{sample}</pre>")
                }, step=step)

            model.train()
        
        step += 1
    
    os.makedirs("chkpts", exist_ok=True)

    # handle torch.compile wrapper
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    torch.save({
        "model": raw_model.state_dict(),
        "optimizer": [opt_adam.state_dict(),opt_muon.state_dict()],
        "scheduler": [scheduler_adam.state_dict(),scheduler_muon.state_dict()],
        "scaler": scaler.state_dict(),
        "step": step,
        "config": config,
    }, "chkpts/pretrain_latest.pt")
        

           


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = TikTokenizer()
    model = TransformerLM(tokenizer.vocab_size, max_len=config["max_len"]).to(device)
    model = torch.compile(model, dynamic=False)
    loader = getMiniPileDataloder(tokenizer, batch_size=config["bs"], max_length=config["max_len"])

    print("=== PRETRAIN ===")
    pretrain(model, loader, tokenizer, device)
    


if __name__ == "__main__":
    main()