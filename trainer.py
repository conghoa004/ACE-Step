from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from datetime import datetime
import argparse
import torch
import json
import os
from pytorch_lightning.core import LightningModule
from torch.utils.data import DataLoader
from acestep.pipeline_ace_step import ACEStepPipeline
from acestep.text2music_dataset import Text2MusicDataset
from transformers import AutoModel, Wav2Vec2FeatureExtractor
import torchaudio
import torch.nn.functional as F
from tqdm import tqdm

torch.backends.cudnn.benchmark = False

class Pipeline16GB(LightningModule):
    def __init__(self, learning_rate=1e-4, num_workers=4, dataset_path="./data", lora_config_path=None, adapter_name="lora_adapter", max_steps=200000):
        super().__init__()
        self.save_hyperparameters()
        # Load ACEStep pipeline
        self.ace_pipeline = ACEStepPipeline(checkpoint_dir=None)
        self.ace_pipeline.load_checkpoint(self.ace_pipeline.checkpoint_dir)

        # Load transformer, half precision + GPU
        self.transformers = self.ace_pipeline.ace_step_transformer.half().cuda()
        self.transformers.enable_gradient_checkpointing()

        # LoRA adapter
        if lora_config_path is not None:
            from peft import LoraConfig
            with open(lora_config_path, encoding="utf-8") as f:
                lora_cfg_json = json.load(f)
            lora_config = LoraConfig(**lora_cfg_json)
            self.transformers.add_adapter(adapter_config=lora_config, adapter_name=adapter_name)
            self.adapter_name = adapter_name

        # Freeze full model
        for p in self.transformers.parameters():
            p.requires_grad = False
        # Only LoRA params trainable
        for p in self.transformers.get_adapter(self.adapter_name).parameters():
            p.requires_grad = True

        # DCAE & text encoder CPU
        self.dcae = self.ace_pipeline.music_dcae.float().cpu()
        self.text_encoder_model = self.ace_pipeline.text_encoder_model.float().cpu()
        self.text_tokenizer = self.ace_pipeline.text_tokenizer
        self.dcae.requires_grad_(False)
        self.text_encoder_model.requires_grad_(False)

        # MERT & mHuBERT CPU
        self.mert_model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True).eval().cpu()
        self.hubert_model = AutoModel.from_pretrained("utter-project/mHuBERT-147").eval().cpu()
        self.resampler_mert = torchaudio.transforms.Resample(48000, 24000)
        self.resampler_mhubert = torchaudio.transforms.Resample(48000, 16000)
        self.processor_mert = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
        self.processor_mhubert = Wav2Vec2FeatureExtractor.from_pretrained("utter-project/mHuBERT-147")

    def preprocess(self, batch):
        device = torch.device("cuda")
        dtype = torch.float16
        bs = batch["target_wavs"].shape[0]

        # DCAE encode CPU -> GPU
        target_latents, _ = self.dcae.encode(batch["target_wavs"].cpu(), batch["wav_lengths"].cpu())
        target_latents = target_latents.to(device=device, dtype=dtype)

        # Text embeddings
        inputs = self.text_tokenizer(batch["prompts"], return_tensors="pt", padding=True, truncation=True, max_length=256)
        for k in inputs:
            inputs[k] = inputs[k].to(device)
        with torch.no_grad():
            encoder_hidden = self.text_encoder_model(**inputs).last_hidden_state.to(dtype=dtype)

        # Speaker embeddings
        speaker_embds = batch["speaker_embs"].to(device=device, dtype=dtype)
        # Lyrics
        lyric_token_ids = batch["lyric_token_ids"].to(device=device)
        lyric_mask = batch["lyric_masks"].to(device=device)

        # Attention mask
        attention_mask = torch.ones(bs, target_latents.shape[-1], device=device, dtype=dtype)
        return target_latents, attention_mask, encoder_hidden, speaker_embds, lyric_token_ids, lyric_mask

    def training_step(self, batch, batch_idx):
        target_latents, attention_mask, encoder_hidden, speaker_embds, lyric_token_ids, lyric_mask = self.preprocess(batch)

        # Random noise
        noise = torch.randn_like(target_latents, device=target_latents.device)
        noisy = target_latents + noise

        # Forward
        transformer_out = self.transformers(
            hidden_states=noisy,
            attention_mask=attention_mask,
            encoder_text_hidden_states=encoder_hidden,
            speaker_embeds=speaker_embds,
            lyric_token_idx=lyric_token_ids,
            lyric_mask=lyric_mask,
            timestep=torch.zeros(target_latents.shape[0], device=target_latents.device, dtype=target_latents.dtype)
        ).sample

        loss = F.mse_loss(transformer_out, target_latents)
        self.log("train/loss", loss, on_step=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        trainable_params = [p for p in self.transformers.get_adapter(self.adapter_name).parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=self.hparams.learning_rate)
        return optimizer

    def train_dataloader(self):
        dataset = Text2MusicDataset(train=True, train_dataset_path=self.hparams.dataset_path)
        return DataLoader(dataset, shuffle=True, num_workers=self.hparams.num_workers, pin_memory=True, collate_fn=dataset.collate_fn)

def main(args):
    model = Pipeline16GB(
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        dataset_path=args.dataset_path,
        lora_config_path=args.lora_config_path,
        adapter_name=args.exp_name,
        max_steps=args.max_steps
    )
    checkpoint_callback = ModelCheckpoint(every_n_train_steps=args.every_n_train_steps, save_top_k=-1)
    logger_callback = TensorBoardLogger(version=datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + args.exp_name, save_dir=args.logger_dir)
    trainer = Trainer(
        accelerator="gpu",
        devices=args.devices,
        precision=16,
        accumulate_grad_batches=args.accumulate_grad_batches,
        max_steps=args.max_steps,
        log_every_n_steps=1,
        logger=logger_callback,
        callbacks=[checkpoint_callback],
        gradient_clip_val=args.gradient_clip_val
    )
    trainer.fit(model, ckpt_path=args.ckpt_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dataset_path", type=str, default="./zh_lora_dataset")
    parser.add_argument("--exp_name", type=str, default="chinese_rap_lora")
    parser.add_argument("--lora_config_path", type=str, default="config/zh_rap_lora_config.json")
    parser.add_argument("--every_n_train_steps", type=int, default=2000)
    parser.add_argument("--logger_dir", type=str, default="./exps/logs/")
    parser.add_argument("--accumulate_grad_batches", type=int, default=2)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--gradient_clip_val", type=float, default=0.5)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=200000)
    args = parser.parse_args()
    main(args)
