import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from datetime import datetime
from transformers import AutoModel, Wav2Vec2FeatureExtractor
import torchaudio
import os
import random
from acestep.pipeline_ace_step import ACEStepPipeline
from acestep.text2music_dataset import Text2MusicDataset
from acestep.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from peft import LoraConfig
from loguru import logger
from diffusers.utils.torch_utils import randn_tensor
from acestep.apg_guidance import apg_forward, MomentumBuffer
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from tqdm import tqdm

torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision("high")


class Pipeline(LightningModule):
    def __init__(self, dataset_path, checkpoint_dir, lora_config_path, adapter_name="lora_adapter",
                 learning_rate=1e-4, max_steps=200000, num_workers=4, T=1000, ssl_coeff=1.0):
        super().__init__()
        self.save_hyperparameters()
        self.is_train = True
        self.T = T

        # Load ACE-Step pipeline checkpoint
        acestep_pipeline = ACEStepPipeline(checkpoint_dir)
        acestep_pipeline.load_checkpoint(checkpoint_dir)

        # Load transformer -> GPU half precision
        self.transformers = acestep_pipeline.ace_step_transformer.half().cuda()
        self.transformers.enable_gradient_checkpointing()

        # LoRA
        if lora_config_path is not None:
            with open(lora_config_path, encoding="utf-8") as f:
                lora_cfg_dict = json.load(f)
            lora_cfg = LoraConfig(**lora_cfg_dict)
            self.transformers.add_adapter(adapter_config=lora_cfg, adapter_name=adapter_name)
            self.adapter_name = adapter_name

        # Offload large models to CPU
        self.dcae = acestep_pipeline.music_dcae.float().cpu()
        self.dcae.requires_grad_(False)

        self.text_encoder_model = acestep_pipeline.text_encoder_model.float().cpu()
        self.text_encoder_model.requires_grad_(False)
        self.text_tokenizer = acestep_pipeline.text_tokenizer

        # SSL models on CPU
        self.mert_model = AutoModel.from_pretrained(
            "m-a-p/MERT-v1-330M", trust_remote_code=True
        ).eval().cpu()
        self.hubert_model = AutoModel.from_pretrained("utter-project/mHuBERT-147").eval().cpu()

        self.resampler_mert = torchaudio.transforms.Resample(48000, 24000)
        self.processor_mert = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M")

        self.resampler_mhubert = torchaudio.transforms.Resample(48000, 16000)
        self.processor_mhubert = Wav2Vec2FeatureExtractor.from_pretrained("utter-project/mHuBERT-147")

        self.ssl_coeff = ssl_coeff
        self.scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=self.T, shift=3.0)

    def configure_optimizers(self):
        trainable_params = [p for n, p in self.transformers.named_parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=self.hparams.learning_rate, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0.0, 1 - step / self.hparams.max_steps))
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def train_dataloader(self):
        dataset = Text2MusicDataset(train=True, train_dataset_path=self.hparams.dataset_path)
        return DataLoader(dataset, batch_size=1, shuffle=True, num_workers=self.hparams.num_workers, collate_fn=dataset.collate_fn)

    def preprocess(self, batch):
        device = "cuda"
        dtype = torch.float16
        bs = batch["target_wavs"].shape[0]

        # Text embeddings on CPU -> move to GPU
        texts = batch["prompts"]
        inputs = self.text_tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        text_hidden = self.text_encoder_model(**inputs).last_hidden_state.to(device=device, dtype=dtype)
        attention_mask = inputs["attention_mask"].to(device)

        # Target latents -> DCAE CPU -> GPU half
        target_latents, _ = self.dcae.encode(batch["target_wavs"], batch["wav_lengths"])
        target_latents = target_latents.to(device=device, dtype=dtype)
        attention_mask_latent = torch.ones(bs, target_latents.shape[-1], device=device, dtype=dtype)

        # Speaker & lyrics
        speaker_embds = batch["speaker_embs"].to(device=device, dtype=dtype)
        lyric_token_ids = batch["lyric_token_ids"].to(device=device)
        lyric_mask = batch["lyric_masks"].to(device=device)

        return target_latents, attention_mask_latent, text_hidden, attention_mask, speaker_embds, lyric_token_ids, lyric_mask

    def training_step(self, batch, batch_idx):
        device = "cuda"
        dtype = torch.float16
        target_latents, attention_mask, text_hidden, text_attention_mask, speaker_embds, lyric_token_ids, lyric_mask = self.preprocess(batch)

        noise = torch.randn_like(target_latents, device=device, dtype=dtype)
        timesteps = self.scheduler.timesteps[:target_latents.shape[0]].to(device=device, dtype=dtype)
        sigmas = self.scheduler.sigmas[:target_latents.shape[0]].to(device=device, dtype=dtype)
        noisy_latents = sigmas * noise + (1.0 - sigmas) * target_latents

        # Forward transformer
        transformer_out = self.transformers(
            hidden_states=noisy_latents,
            attention_mask=attention_mask,
            encoder_text_hidden_states=text_hidden,
            text_attention_mask=text_attention_mask,
            speaker_embeds=speaker_embds,
            lyric_token_idx=lyric_token_ids,
            lyric_mask=lyric_mask,
            timestep=timesteps
        )
        pred = transformer_out.sample
        loss = F.mse_loss(pred, target_latents)
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        return loss


def run_training(args):
    model = Pipeline(
        dataset_path=args.dataset_path,
        checkpoint_dir=args.checkpoint_dir,
        lora_config_path=args.lora_config_path,
        adapter_name=args.exp_name,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        num_workers=args.num_workers,
    )

    checkpoint_callback = ModelCheckpoint(every_n_train_steps=args.every_n_train_steps, save_top_k=-1)
    logger_callback = TensorBoardLogger(save_dir=args.logger_dir,
                                        version=datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + args.exp_name)

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        precision=16,
        accumulate_grad_batches=args.accumulate_grad_batches,
        max_steps=args.max_steps,
        callbacks=[checkpoint_callback],
        logger=logger_callback,
        gradient_clip_val=args.gradient_clip_val
    )
    trainer.fit(model)


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="./zh_lora_dataset")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--lora_config_path", type=str, default="config/zh_rap_lora_config.json")
    parser.add_argument("--exp_name", type=str, default="chinese_rap_lora")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=200000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--every_n_train_steps", type=int, default=2000)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--gradient_clip_val", type=float, default=0.5)
    args = parser.parse_args()

    run_training(args)
