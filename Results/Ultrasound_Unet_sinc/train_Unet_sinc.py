import os
import numpy as np
import sys
import torch
import logging
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml
from pathlib import Path
import torchaudio
from scipy.io import loadmat
from speechbrain.utils.parameter_transfer import Pretrainer
from SharedFuncs import *

logger = logging.getLogger(__name__)


class Ultra_Brain(sb.Brain):
    def compute_forward(self, batch):
        #print('START')
        batch = batch.to(self.device)
        rf = batch.sig.data # removing the the length flag of the PaddedData type
        
        
        
        # ### Normalization of input
        # batch_size, height = rf.shape # shape [batch,timeseries]
        # rf = rf.view(rf.size(0), -1)
        # rf -= rf.min(1, keepdim=True)[0]
        # rf /= rf.max(1, keepdim=True)[0]
        # rf = rf.view(batch_size, height)


        # Mean Normalization
        norm = sb.processing.features.InputNormalization()
        rf = features = norm(batch.sig.data,batch.sig.lengths)
        rf = rf.type(torch.cuda.FloatTensor)
        rf_unsqueeze = rf.unsqueeze(dim=1)

        ## SincConv Does not neet Extra channel!!
        a1 = self.modules.SincBlock(rf)
        a1 = torch.transpose(a1, 1, 2)
        print('A1 Size', a1.shape)

        a2 = self.modules.UPipe(rf_unsqueeze)
        a = torch.cat((a1, a2), 2)
        print('a SHAPE', a.shape)
        a = self.modules.RestPipe(a)
        logits = self.modules.MLPBlock(a)
        
        #print('OUT',logits)

        return logits 

    def compute_objectives(self, predictions, batch):
        #print('PREDICTION', predictions.shape, batch.att.shape )
        attenuation = batch.att
        attenuation = attenuation.type(torch.cuda.FloatTensor)
        return sb.nnet.losses.mse_loss(predictions, attenuation.unsqueeze(1))
    
    def fit_batch(self, batch):
        predictions = self.compute_forward(batch)
        #predictions = predictions.squeeze()
        #print('PREDICTION', predictions.shape, batch.att.shape )
        loss = self.compute_objectives(predictions, batch)
        loss.backward()
        if self.check_gradients(loss):
            self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.detach()

    def evaluate_batch(self, batch,stage):
        # if stage == sb.Stage.TEST:
        #     print('BATCH shape',batch.sig.data.shape)
        if stage == sb.Stage.VALID or stage == sb.Stage.TEST:
            predictions = self.compute_forward(batch)
            with torch.no_grad():
                loss = self.compute_objectives(predictions, batch)
                #print("EVALUATE BATCH loss", loss)
            return loss.detach()
        
    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Store the train loss until the validation stage.
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats


        # Perform end-of-iteration things, like annealing, logging, etc.
        elif stage == sb.Stage.VALID:
            

            # Update learning rate
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats={
                    "loss": stage_loss,
                },
            )

            # Save the current checkpoint and delete previous checkpoints.
            self.checkpointer.save_and_keep_only(
                meta={"loss": stage_stats["loss"]}, min_keys=["loss"],
            )

        # We also write statistics about test data to stdout and to the logfile.
        elif stage == sb.Stage.TEST:
            
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats={
                    "loss": stage_loss,
                },
            )
                






if __name__ == "__main__":

    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    print(hparams_file, run_opts, overrides)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
        print(hparams)
    
    sb.create_experiment_directory(
    experiment_directory=hparams["output_folder"],
    hyperparams_to_save=hparams_file,
    overrides=overrides,
    )

    train_data,  valid_data, test_data = dataio_prepare(hparams)
    
    Ultra_brain = Ultra_Brain(
    modules=hparams["modules"],
    opt_class=hparams["optim"],
    hparams=hparams,
    run_opts=run_opts,
    checkpointer=hparams["checkpointer"],
    )

    Ultra_brain.fit(
    Ultra_brain.hparams.epoch_counter,
    train_data,
    valid_data,
    train_loader_kwargs=hparams["train_dataloader_opts"],
    valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    test_stats = Ultra_brain.evaluate(
        test_set=test_data,
        test_loader_kwargs=hparams["test_dataloader_opts"],
        progressbar = True
    )

    training_losses , validation_losses = get_losses(hparams["train_log"])

    plt.plot(validation_losses, label='sinc_Unet_validation')
    plt.plot(training_losses, label='sinc_Unet_training')
    plt.ylabel('Loss')
    plt.xlabel('# Epochs')
    plt.legend()
    plt.xticks(range(1,len(validation_losses)+1))
    plt.savefig(os.path.join(hparams['loss_image_folder'],'sinc_Unet_epoch_'+ str(hparams['number_of_epochs'])+
                 '_batchsize_'+str(hparams['batch_size'])+
                 '_ChanellNum_'+str(hparams['CHANNEL_NUM'])+
                 '_Shufelling_'+str(hparams['sorting'])+'.png'))
    #plt.show()