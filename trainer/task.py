
from typing import Any, Optional

from lightning.pytorch.loggers import WandbLogger
import lightning as L
from litdata import StreamingDataset, StreamingDataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from huggingface_hub import PyTorchModelHubMixin

from torch import nn
import os
import glob 

os.environ["WANDB_API_KEY"] = (
    'your api key'
)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = './path-to-credentials/.json'


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class RotaryPositionalEmbeddings(nn.Module):
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864.

    Reference implementation (used for correctness verfication)
    can be found here:
    https://github.com/meta-llama/llama/blob/689c7f261b9c5514636ecc3c5fefefcbb3e6eed7/llama/model.py#L132

    In this implementation we cache the embeddings for each position upto
    ``max_seq_len`` by computing this during init.

    Args:
         dim (int): Embedding dimension. This is usually set to the dim of each
              head in the attention module computed as ``embed_dim // num_heads``
         max_seq_len (int): Maximum expected sequence length for the
              model, if exceeded the cached freqs will be recomputed
         base (int): The base for the geometric progression used to compute
              the rotation angles
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10_000,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.rope_init()

    def rope_init(self):
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(
        self, x: torch.Tensor, *, input_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
             x (torch.Tensor): input tensor with shape
                  ``[b, s, n_h, h_d]``
             input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                  of each token. During training, this is used to indicate the positions
                  of each token relative to its sample when packed, shape [b, s].
                  During inference, this indicates the position of the current token.
                  If none, assume the index of the token is its position id. Default is None.

        Returns:
             torch.Tensor: output tensor with shape ``[b, s, n_h, h_d]``

        Notation used for tensor shapes:
             - b: batch size
             - s: sequence length
             - n_h: num heads
             - h_d: head dim
        """
        # input tensor has shape [b, s, n_h, h_d]
        seq_len = x.size(1)

        # extract the values based on whether input_pos is set or not
        rope_cache = (
            self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        )

        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        # tensor has shape [b, s, n_h, h_d // 2, 2]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # reshape the cache for broadcasting
        # tensor has shape [b, s, 1, h_d // 2, 2] if packed samples,
        # otherwise has shape [1, s, 1, h_d // 2, 2]
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        # tensor has shape [b, s, n_h, h_d // 2, 2]
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        # tensor has shape [b, s, n_h, h_d]
        x_out = x_out.flatten(3)
        return x_out.type_as(x)


class Attention_Head(nn.Module):
    def __init__(self, embed_dims, head_size, num_heads):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_size = head_size
        self.total_heads = head_size * num_heads

        self.q_proj = nn.Linear(embed_dims, self.total_heads)
        self.k_proj = nn.Linear(embed_dims, self.total_heads)
        self.v_proj = nn.Linear(embed_dims, self.total_heads)
        self.o_proj = nn.Linear(self.total_heads, embed_dims)
        self.pe = RotaryPositionalEmbeddings(self.head_size)

    def forward(self, logits, batch_size, seq_len):
          q = self.q_proj(logits).view(batch_size, seq_len, self.num_heads, self.head_size)
          k = self.k_proj(logits).view(batch_size, seq_len, self.num_heads, self.head_size)

          # Apply RoPE while shape is [b, s, n_h, h_d] ✓
          q_pe = self.pe(q).transpose(1, 2)
          k_pe = self.pe(k).transpose(1, 2)

          v = (
               self.v_proj(logits)
               .view(batch_size, seq_len, self.num_heads, self.head_size)
               .transpose(1, 2)
          )

          attention_out = F.scaled_dot_product_attention(q_pe, k_pe, v, is_causal=True)
          out = (
               attention_out.transpose(1, 2)
               .contiguous()
               .view(batch_size, seq_len, self.total_heads)
          )
          return self.o_proj(out)


class FFN(nn.Sequential):
    def __init__(self, embed_dims, hidden_dim):
        super().__init__(
            nn.Linear(embed_dims, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dims),
        )


class Block(nn.Module):
    def __init__(self, embed_dims, head_size, num_heads):
        super().__init__()
        self.embed_dims = embed_dims
        self.head_size = head_size

        self.rms_Norm1 = RMSNorm(embed_dims)
        self.rms_Norm2 = RMSNorm(embed_dims)

        self.Attention_Head = Attention_Head(embed_dims, head_size, num_heads)
        self.FFN = FFN(embed_dims, embed_dims * 4)

    def forward(self, logits, batch_size, seq_len):
        x = self.Attention_Head(self.rms_Norm1(logits), batch_size, seq_len)
        x = x + logits
        out = self.FFN(self.rms_Norm2(x))
        out = out + x
        return out


class LightningTransformer(L.LightningModule, PyTorchModelHubMixin):
    def __init__(
        self,
        batch_size,
        seq_len,
        embed_dims,
        head_size,
        num_heads,
        block_num,
        vocab_size,
        lr,
        iterations,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.embed_dims = embed_dims
        self.head_size = head_size
        self.num_heads = num_heads
        self.vocab_size = vocab_size

        self.lr = lr
        self.iterations = iterations

        self.token_embed = nn.Embedding(vocab_size, embed_dims)
        self.rms_Norm_embed = RMSNorm(embed_dims)
        self.embed_proj = nn.Linear(embed_dims, vocab_size)
        self.block_list = nn.ModuleList(
            [Block(embed_dims, head_size, num_heads) for _ in range(block_num)]
        )

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            self.lr,
            total_steps=self.iterations,
            pct_start=0.1,
            anneal_strategy="cos",
            final_div_factor=100.0,  # iterations must be divisible by 100
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self(x, y)
        self.log("train_loss", loss)
        return loss

    def forward(self, inputs, target=None):
        batch_size, seq_len = inputs.shape
        logits = self.token_embed(inputs)

        for block in self.block_list:
            logits = block(logits, batch_size, seq_len)

        unembed_out = self.embed_proj(self.rms_Norm_embed(logits))

        if target is not None:
            preds = unembed_out.view(batch_size * seq_len, -1)
            target = target.view(-1)

            loss_fn = F.cross_entropy(preds, target)
            return loss_fn

        return unembed_out

    def generate(self, input_tokens, max_tokens):
        for _ in range(max_tokens):
            last_seq = input_tokens[:, -self.seq_len :]
            logits = self(last_seq)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            input_tokens = torch.cat((input_tokens, next_tok), dim=1)
        return input_tokens


class LitdataStreamingDataset(StreamingDataset):
    def __getitem__(self, idx):
        chunk = super().__getitem__(idx)
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


class LightningDataLoader(L.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        dataset = LitdataStreamingDataset(
            input_dir=self.data_dir,
            shuffle=True,
        )
        return StreamingDataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
        )


# GCS_PREV_MODEL_DIR = (
#     "gs://apollousa-sg-models/apollousa-670M/training/model-step-40000.ckpt"
# )
GCS_MODEL_DIR = "/gcs/apollousa-sg-models/apollousa-670M/training/v2"
GCS_DATASET_BUCKET = "/gcs/apollousa-sg-datasets"

batch_size = 8
seq_len = 1024
embed_dims = 1152
head_size = 96
num_heads = 12
block_num = 33
vocab_size = 65536
lr = 4e-4
iterations = 100000

num_workers = 8

def main():
    wandb_logger = WandbLogger(
         log_model="all",
         resume="allow",
         id="apollousa-670m-v2"
     )

    model = LightningTransformer(
        batch_size=batch_size,
        seq_len=seq_len,
        embed_dims=embed_dims,
        head_size=head_size,
        num_heads=num_heads,
        block_num=block_num,
        vocab_size=vocab_size,
        lr=lr,
        iterations=iterations,
    )

    dataloader = LightningDataLoader(
        data_dir=GCS_DATASET_BUCKET,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    trainer = L.Trainer(
        logger=wandb_logger,
        max_epochs=1,
        limit_train_batches=iterations,
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        accumulate_grad_batches=4,
        log_every_n_steps=10,
        enable_checkpointing=True,
        devices=1,
        strategy="auto",
        callbacks=[
            L.pytorch.callbacks.ModelCheckpoint(
               dirpath=GCS_MODEL_DIR, every_n_train_steps=500, save_top_k=-1
            )
        ],
    )
    checkpoints = sorted(glob.glob("/gcs/apollousa-sg-models/apollousa-670M/training/v2/*.ckpt"))
    latest_ckpt = max(checkpoints, key=os.path.getmtime) if checkpoints else None
    
    trainer.fit(model, datamodule=dataloader, ckpt_path=latest_ckpt)
    
if __name__ == '__main__':
    main()

""" 
looking for:
- wandb logs
- vertex ai training pipeline custom job shown in ui
- model weights saving in apollousa-sg-models bucket
- tracking via vertex ai ???

warnings:
- 3.10 python currently used soon to be deprecated for vertex ai

"""
