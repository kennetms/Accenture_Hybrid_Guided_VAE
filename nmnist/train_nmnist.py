#!/bin/python
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../model')
sys.path.insert(1, '../')
from train_hybrid_vae_guided_base import Guide, HybridGuidedVAETrainer
import matplotlib
matplotlib.use('Agg')
from hybrid_beta_vae import Reshape, VAE
from decolle.utils import parse_args, train, test, accuracy, save_checkpoint, load_model_from_checkpoint, prepare_experiment, write_stats, cross_entropy_one_hot, MultiOpt
#from utils import save_checkpoint, load_model_from_checkpoint
import datetime, os, socket, tqdm
import numpy as np
import torch
from torch import nn
import importlib
from itertools import chain
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchneuromorphic import transforms
from tqdm import tqdm
import math
import sys
from utils import generate_process_target
import argparse

epsilon = sys.float_info.epsilon
np.set_printoptions(precision=4)



class NMNISTTrainer(HybridGuidedVAETrainer):
    """
        Subclass of HybridGuidedVAETrainer that modifies
        relevant functions to make it friendly for training
        models with the NMNIST dataset
    """
    def __init__(self, param_file, dataset_path):
        
        super(NMNISTTrainer, self).__init__(param_file, dataset_path)
        
        
    def generate_data_batch(self, dataset, dataset_path):
        """
            Generates a batch of data to ensure the data loader is working
            as well as providing some data to plot images for comparison
            to reconstructed images
            
            inputs:
                - NeuromorphicDataset dataset: A torchneuromorphic dataloader
                - string dataset_path: The path to the .hdf5 file containing the data to load from
                
        """
        try:
            create_data = dataset.create_data
        except AttributeError:
            create_data = dataset.create_dataloader

        self.train_dl, self.test_dl = create_data(root=dataset_path,
                                          chunk_size_train=self.params['chunk_size_train'],
                                          chunk_size_test=self.params['chunk_size_test'],
                                          batch_size=self.params['batch_size'],
                                          dt=self.params['deltat'],
                                          num_workers=self.params['num_dl_workers'])
        self.data_batch, self.target_batch = next(iter(self.train_dl))

        self.data_batch = torch.Tensor(self.data_batch).to(self.device)
        self.target_batch = torch.Tensor(self.target_batch).to(self.device)
    
    
    def train_guided_aug(self):
        """
            The training loop for training the hybrid guided VAE model.
            Iterates through data batches loaded from the dataloader
            and inputs them into self.train_step_guided() for learning
            
            outputs:
                - float loss_batch: the loss of the vae model
                - float excite_batch: the loss of the excititory guide classifier
                - float inhib_batch: the loss of the inhibitory guide classifier
                - float abs_batch: absolute difference of the excititory and inhibitory losses
                - float entropy_batch: 
                - float mean_batch: mean of the losses
        """
        self.net.train()
        loss_batch = []
        excite_batch = []
        inhib_batch = []
        abs_batch = []
        entropy_batch = []
        mean_batch = []
        for i in range(self.params['num_augs']):
            for x,t in tqdm(iter(self.train_dl)):
                new_t = t[t[:,-1,:].argmax(1)!=10]
                new_t = new_t[:,-1,:].argmax(1)
                #print(new_t.shape)
                x = x[t[:,-1,:].argmax(1)!=10]
                x_c = x.to(self.device)
                frames = self.process_target(x_c,i-1)
                loss_, excite_loss_, inhib_loss_, loss_abs_, soft_entropy, soft_mean = self.train_step_guided(x_c,frames.to(self.device),new_t.long(),self.params['vae_beta'])
                loss_batch.append(loss_.detach().cpu().numpy())
                excite_batch.append(excite_loss_.detach().cpu().numpy())
                inhib_batch.append(inhib_loss_.detach().cpu().numpy())
                abs_batch.append(loss_abs_.detach().cpu().numpy())
                entropy_batch.append(soft_entropy)
                mean_batch.append(soft_mean)#.detach().cpu().numpy())
        return np.mean(loss_batch,dtype=np.float64), np.mean(excite_batch,dtype=np.float64), np.mean(inhib_batch,dtype=np.float64), np.mean(abs_batch,dtype=np.float64), np.mean(entropy_batch,dtype=np.float64), np.mean(mean_batch,dtype=np.float64)
    
    
    def get_latent_space(self, dl):
        """
            Inputs data generated from the dataloader
            into the VAE encoder to output the corresponding latent space
            and returns the latent space with the corresponding targets.
            
            inputs:
                - one of self.train_dl or self.test_dl
                
            outputs:
                - ndarray lats: The latent space for each datapoint in the dataset
                - ndarray tgts: The targets corresponding to each datapoint in the dataset
        """
        #all_d = []
        lats = []
        tgts = []

        for x,t in tqdm(iter(dl)):

            new_t = t[t[:,-1,:].argmax(1)!=10]
            if new_t.shape[0] == 0:
                continue
            new_t = new_t[:,-1,:].argmax(1)
            #print(new_t.shape)
            x = x[t[:,-1,:].argmax(1)!=10]

            with torch.no_grad():
                mu, logvar = self.net.encode(x.to(self.device))
                lat = self.net.reparameterize(mu,logvar).detach().cpu().numpy()
                lats += lat.tolist()
                tgts += new_t.tolist()

                    #all_d += process_target(x).tolist()
        return np.array(lats), np.array(tgts)#[:,-1,:].argmax(1)
    
    
    def tsne_project(self, lats, tgts, do_plot = True):
        """
            Calculates tsne projections of the latent space
            and color codes them according to the provided targets
            for visualization of how close or far data points are in the reduced tsne projection
            to get a sense of how well the model is disentangling the data in the latent space
            
            inputs:
                - ndarray lats: The latent space for each datapoint in the dataset
                - ndarray tgts: The targets corresponding to each datapoint in the dataset
                - bool do_plot: If true, plots the tsnes and returns them as figures
                                If False just returns the tsne projection
            outputs:
                - ndarray lat_tsne: The tsne projection of the latent space
                - pyplot figure fig: The plot of the tsne projection of the latent space
                - pyplot figure fig2:
                - pyplot figure fig3:
                - pyplot figure fig4: The plot of the tsne projection of the excititory portion of the latent space
                - pyplot figure fig5: The plot of the tsne projection of the inhibitory portion of the latent space
        """
        from sklearn.manifold import TSNE
        #print(lats.shape)
        lat_tsne = TSNE(n_components=2).fit_transform(lats)
        inhib_tsne = TSNE(n_components=2).fit_transform(self.inhib.inhibit_z(torch.from_numpy(lats)).numpy())
        exc_tsne = TSNE(n_components=2).fit_transform(self.inhib.excite_z(torch.from_numpy(lats)).numpy())
        tgts = tgts[:10]
        if do_plot:
            fig = plt.figure(figsize=(16,10))
            fig2 = plt.figure(figsize=(16,10))
            fig3 = plt.figure(figsize=(16,10))
            fig4 = plt.figure(figsize=(16,10))
            fig5 = plt.figure(figsize=(16,10))
            ax = fig.add_subplot()
            ax4 = fig4.add_subplot()
            ax5 = fig5.add_subplot()

            for i in range(params['num_classes']):#1):
                idx = tgts==i
                ax.scatter(lat_tsne[idx,0],lat_tsne[idx,1], label = dataset.mapping[i])
                ax4.scatter(exc_tsne[idx,0],exc_tsne[idx,1], label = dataset.mapping[i])
                ax5.scatter(inhib_tsne[idx,0],inhib_tsne[idx,1], label = dataset.mapping[i])
            ax.legend()
            ax4.legend()
            ax5.legend()

            return lat_tsne, fig, fig2, fig3, fig4, fig5
        else:
            return lat_tsne
        
        
    def train_eval_plot_loop(self):
        """
            The main function more or less.
            Runs all of the training  and testing epochs in loops
            Gets and plots latent spaces as tsnes, latent traversals, and reconstructions
            saves checkpoints to view in tensorboard and for loading models later for additional training or inference
        """
        
        if not self.args.no_train:
            orig = self.process_target(self.data_batch).detach().cpu().view(*[[-1]+self.params['output_shape']])[:,0:1]
            figure2 = plt.figure(99)
            plt.imshow(make_grid(orig, scale_each=True, normalize=True).transpose(0,2).numpy())
            self.writer.add_figure('original_train',figure2,global_step=1)

            for e in tqdm(range(self.starting_epoch , self.params['num_epochs'] )):
                interval = e // self.params['lr_drop_interval']
                for i,opt_ in enumerate(self.opt):
                    lr = self.opt.param_groups[-1]['lr']
                    if interval > 0:
                        print('Changing learning rate from {} to {}'.format(lr, opt_.param_groups[-1]['lr']))
                        opt_.param_groups[-1]['lr'] = np.array(self.params['learning_rate'][i]) / (interval * self.params['lr_drop_factor'])
                    else:
                        print('Changing learning rate from {} to {}'.format(lr, opt_.param_groups[-1]['lr']))
                        opt_.param_groups[-1]['lr'] = np.array(self.params['learning_rate'][i])

                if (e % self.params['test_interval']) == 0 and e!=0:
                    print('---------------Epoch {}-------------'.format(e))
                    if not self.args.no_save:
                        print('---------Saving checkpoint---------')
                        save_checkpoint(e, self.checkpoint_dir, self.net, self.opt, self.net.cls_sq, self.inhib)

                    #test here

                    # tsne
                    lats, tgts = self.get_latent_space(self.train_dl)
                    lats_test, tgts_test = self.get_latent_space(self.test_dl)

                    #latent space traversal
                    fig = self.latent_traversal(lats, tgts, 1)

                    fig_test = self.latent_traversal(lats_test, tgts_test, 1)

                    fig_switch = self.latent_traversal_switch(lats, tgts, 1, 2)

                    fig_inhib = self.latent_traversal_inhib(lats, tgts, 1)

                    # commented out because this takes a long time to process
                    #_, figure, fig2, fig3, fig6, fig8 = tsne_project(lats, tgts)
                    #_, figure2, fig4, fig5, fig7, fig9 = tsne_project(lats_test, tgts_test)

                    self.writer.add_figure('latent_traversal',fig,global_step=e)
                    self.writer.add_figure('latent_traversal_test',fig_test,global_step=e)
                    self.writer.add_figure('latent_traversal_switch',fig_switch,global_step=e)
                    self.writer.add_figure('latent_traversal_inhib',fig_inhib,global_step=e)
                    #writer.add_figure('tsne_train',figure,global_step=e)
                    #writer.add_figure('tsne_test',figure2,global_step=e)
                    #writer.add_figure('exc_train',fig6,global_step=e)
                    #writer.add_figure('exc_test',fig7,global_step=e)

                    # reconstruction
                    recon_batch, mu, logvar, clas = self.net(self.data_batch.to(self.device))
                    recon_batch_c = recon_batch.detach().cpu()
                    figure = plt.figure()
                    img = recon_batch_c.view(*[[-1]+self.params['output_shape']])[:,0:1]
                    print(img.shape)
                    plt.imshow(make_grid(img, scale_each=True, normalize=True).transpose(0,2).numpy())
                    self.writer.add_figure('recon_train',figure,global_step=e)

                #train_here

                loss_, excite_loss_, inhib_loss_, loss_abs_, entropy, means = self.train_guided_aug()
                self.writer.add_scalar('inhibitory_net_loss_1', excite_loss_, e)
                self.writer.add_scalar('inhibitory_net_loss_2', inhib_loss_, e)
                self.writer.add_scalar('clas_loss', loss_abs_, e)
                #writer.add_scalar('entropy', entropy, e)
                #writer.add_scalar('mean_values_softmax_inp', means, e)

                lats, tgts = self.get_latent_space(self.train_dl)
                lats_test, tgts_test = self.get_latent_space(self.test_dl)

                train_acc = self.eval_accuracy(lats, tgts, True)
                test_acc = self.eval_accuracy(lats_test, tgts_test, True)
                self.writer.add_scalar('vaeclas_net_train_acc', train_acc, e)
                self.writer.add_scalar('vaeclas_net_test_acc', test_acc, e)

                self.writer.add_scalar('train_loss', loss_, e)

                plt.close('all') # close figures so they don't use too much memory...


if __name__=="__main__":
    # parse args for params file, and dataset_path, that should take care of pretty much everything I think...
    # I might need to add something for lights, but I'll figure it out

    parser = argparse.ArgumentParser('HGVAE')
    
    parser.add_argument('--params-file', default = '../parameters/params_hybridvae_nmnist-guidedbeta-Copy1.yml', type=str, help='Path to the parameter config file.') 
    parser.add_argument('--data-file', default = '/home/kennetms/Documents/data/n_mnist.hdf5', type=str, help='Path to the file the data is in, should be hdf5 compatible with torchneuromorphic.')
    args = parser.parse_args()
    
    param_file = args.params_file #'parameters/params_hybridvae_dvsgestures-guidedbeta-noaug-Copy1.yml'
    dataset_path = args.data_file #'/home/kennetms/Documents/data/dvs_gestures.hdf5'
    
    HGVAE = NMNISTTrainer(param_file, dataset_path)

    HGVAE.train_eval_plot_loop()