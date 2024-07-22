#!/usr/bin/env/python

"""Recipe for training a transformer-based speech enhancement system using discrete SSL audio representations.

To run this recipe:
> python train_discrete_ssl.py hparams/<path-to-config>.yaml

Authors
 * Luca Della Libera 2024
"""

import os
import sys
import warnings

import speechbrain as sb
import torch
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main

from train_encodec import Enhancement as EnhancementEncodec


# To use in configuration files
def len_(SSL_layers, vocab_size):
    return len(SSL_layers) * vocab_size


class Enhancement(EnhancementEncodec):
    @torch.no_grad()
    def sig_to_toks(self, sig, lens):
        # sig: [B, T]
        self.hparams.codec_quantizer.to(self.device).eval()
        toks, _, _ = self.hparams.codec_quantizer(
            sig,
            lens,
            SSL_layers=self.hparams.SSL_layers,
            deduplicates=[False] * len(self.hparams.SSL_layers),
            bpe_tokenizers=[None] * len(self.hparams.SSL_layers),
        )  # [B, N, K]
        return toks

    @torch.no_grad()
    def toks_to_sig(self, toks):
        # toks: [B, N, K]
        self.hparams.codec_vocoder.device = self.device
        self.hparams.codec_vocoder.to(self.device).eval()

        # Add offset for embedding layer
        all_layer_ids = [1, 3, 7, 12, 18, 23]
        offsets = torch.arange(
            0,
            len(all_layer_ids) * self.hparams.vocab_size,
            self.hparams.vocab_size,
            device=self.device,
        )
        offset_idxes = [all_layer_ids.index(x) for x in self.hparams.SSL_layers]
        offsets = offsets[offset_idxes]
        toks = toks + offsets + 1

        # Handle missing codebooks
        if len(self.hparams.SSL_layers) < len(all_layer_ids):
            full_toks = torch.zeros(
                *toks.shape[:2],
                len(all_layer_ids),
                dtype=toks.dtype,
                device=self.device,
            )
            for i, idx in enumerate(offset_idxes):
                full_toks[..., idx] = toks[..., i]
            toks = full_toks

        self.hparams.codec_vocoder.tokenize = False
        sig = self.hparams.codec_vocoder(toks)[:, 0]  # [B, T]
        return sig


if __name__ == "__main__":
    # Command-line interface
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Filter warnings
    warnings.filterwarnings("once")
    warnings.filterwarnings("ignore", module="torch")

    # If --distributed_launch then create ddp_init_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset preparation
    from voicebank_prepare import prepare_voicebank as prepare_data

    prepare_data_kwargs = {
        "data_folder": hparams["data_folder"],
        "save_folder": hparams["save_folder"],
        "splits": hparams["splits"],
        "num_valid_speakers": hparams["num_valid_speakers"],
    }

    # Due to DDP, do the preparation ONLY on the main Python process
    run_on_main(prepare_data, kwargs=prepare_data_kwargs)

    # Create the datasets objects
    from utils import dataio_prepare

    train_data, valid_data, test_data = dataio_prepare(
        debug=run_opts.get("debug", False), **hparams
    )

    # Pretrain the specified modules
    if "pretrainer" in hparams:
        run_on_main(hparams["pretrainer"].collect_files)
        run_on_main(hparams["pretrainer"].load_collected)

    # Use pretrained embeddings
    if hparams["pretrain_embedding"]:
        # See https://github.com/speechbrain/speechbrain/blob/60062c2536e8122253d6ad0e681208f554528950/speechbrain/lobes/models/huggingface_transformers/discrete_ssl.py#L197
        hparams["codec_quantizer"].to(run_opts["device"]).eval()
        embs = []
        for layer_num, vocabulary in zip(
            hparams["codec_quantizer"].ssl_layer_ids,
            hparams["codec_quantizer"].vocabularies,
        ):
            if layer_num not in hparams["SSL_layers"]:
                continue
            embs.append(
                torch.as_tensor(
                    vocabulary, dtype=torch.float32, device=run_opts["device"]
                )
            )
        embs = torch.cat(embs)
        hparams["embedding"].embedding.weight.data.copy_(embs)

    # Log number of parameters/buffers
    codec_params = sum(
        [x.numel() for x in hparams["codec_quantizer"].state_dict().values()]
        + [x.numel() for x in hparams["codec_vocoder"].state_dict().values()]
    )
    model_params = sum(
        [
            x.numel()
            for module in hparams["modules"].values()
            for x in module.state_dict().values()
        ]
    )
    hparams["train_logger"].log_stats(
        stats_meta={
            f"Codec parameters/buffers (M)": f"{codec_params / 1e6:.2f}",
            "Model parameters/buffers (M)": f"{model_params / 1e6:.2f}",
        },
    )

    # Trainer initialization
    brain = Enhancement(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Train
    brain.fit(
        brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_kwargs"],
        valid_loader_kwargs=hparams["valid_dataloader_kwargs"],
    )

    # Test
    brain.hparams.dwer_file = os.path.join(hparams["output_folder"], "dwer.txt")
    brain.evaluate(
        test_data,
        min_key="TER",
        test_loader_kwargs=hparams["test_dataloader_kwargs"],
    )
