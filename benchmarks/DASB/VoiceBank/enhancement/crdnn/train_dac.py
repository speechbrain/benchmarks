#!/usr/bin/env/python

"""Recipe for training a transformer-based speech enhancement system using DAC audio representations.

To run this recipe:
> python train_dac.py hparams/<path-to-config>.yaml

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


class Enhancement(EnhancementEncodec):
    @torch.no_grad()
    def sig_to_toks(self, sig, lens):
        # sig: [B, T]
        self.hparams.codec.to(self.device).eval()
        toks, _ = self.hparams.codec(
            sig[:, None], n_quantizers=self.hparams.num_codebooks
        )  # [B, K, N]
        toks = toks.movedim(-1, -2)  # [B, N, K]
        return toks

    @torch.no_grad()
    def toks_to_sig(self, toks):
        # toks: [B, N, K]
        self.hparams.codec.to(self.device).eval()
        qfeats, _, _ = self.hparams.codec.quantizer.from_codes(
            toks.movedim(-1, -2)  # [B, K, N]
        )
        sig = self.hparams.codec.decode(qfeats)[:, 0]  # [B, T]
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
        # See https://github.com/descriptinc/descript-audio-codec/blob/c7cfc5d2647e26471dc394f95846a0830e7bec34/dac/nn/quantize.py#L200
        toks = torch.arange(hparams["vocab_size"], device=run_opts["device"])
        toks = (
            toks[:, None, None].expand(-1, hparams["num_codebooks"], -1).clone()
        )  # [C, K, 1]
        hparams["codec"].to(run_opts["device"]).eval()
        with torch.no_grad():
            z_q, z_p, _ = hparams["codec"].quantizer.from_codes(toks)
        z_ps = z_p.split(z_p.shape[1] // toks.shape[1], dim=1)  # [C, D, 1] * K
        z_qs = []
        for i, z_p_i in enumerate(z_ps):
            with torch.no_grad():
                z_q_i = (
                    hparams["codec"].quantizer.quantizers[i].out_proj(z_p_i)
                )  # [C, H, 1]
            z_qs.append(z_q_i)
        assert (z_q == sum(z_qs)).all()
        # Embeddings pre-projections: size = 8
        # embs = torch.cat(z_ps)[:, :, 0]
        # Embeddings post-projections: size = 1024
        embs = torch.cat(z_qs)[:, :, 0]  # [CK, H]
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
