#!/usr/bin/env/python

"""Recipe for training a transformer-based speech separation system using SpeechTokenizer audio representations.

To run this recipe:
> python train_speech_tokenizer.py hparams/<path-to-config>.yaml

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

from train_encodec import Separation as SeparationEncodec


class Separation(SeparationEncodec):
    @torch.no_grad()
    def sig_to_toks(self, sig, lens):
        # sig: [B, T]
        self.hparams.codec.to(self.device).eval()
        toks = self.hparams.codec(sig)[
            : self.hparams.num_codebooks
        ]  # [K, B, N]
        toks = toks.movedim(-3, -1)  # [B, N, K]
        return toks

    @torch.no_grad()
    def toks_to_sig(self, toks):
        # toks: [B, N, K]
        self.hparams.codec.to(self.device).eval()
        toks = toks.movedim(-1, -3)  # [K, B, N]
        sig = self.hparams.codec.decode(toks)  # [B, T]
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
    from librimix_prepare import prepare_librimix as prepare_data

    prepare_data_kwargs = {
        "data_folder": hparams["data_folder"],
        "save_folder": hparams["save_folder"],
        "splits": hparams["splits"],
        "num_speakers": hparams["num_speakers"],
        "add_noise": hparams["add_noise"],
        "version": hparams["version"],
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
        # See https://github.com/ZhangXInFD/SpeechTokenizer/blob/a9f88dc72642b600654a62861e34342babae6c71/speechtokenizer/quantization/core_vq.py#L360
        toks = torch.arange(hparams["vocab_size"], device=run_opts["device"])
        toks = (
            toks[None, :, None].expand(hparams["num_codebooks"], -1, -1).clone()
        )  # [K, C, 1]
        hparams["codec"].to(run_opts["device"]).eval()
        embs = []
        for i, indices in enumerate(toks):
            layer = hparams["codec"].model.quantizer.vq.layers[i]
            with torch.no_grad():
                quantized = layer.decode(indices)  # [C, H, 1]
            embs.append(quantized)
        assert (
            hparams["codec"].model.quantizer.decode(toks) == sum(embs)
        ).all()
        embs = torch.cat(embs)[:, :, 0]  # [CK, H]
        hparams["embedding"].embedding.weight.data.copy_(embs)

    # Log number of parameters/buffers
    codec_params = sum(
        [x.numel() for x in hparams["codec"].state_dict().values()]
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
    brain = Separation(
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
