"""Run model inference and save outputs for analysis"""
import os
import hydra
from omegaconf import DictConfig, omegaconf, OmegaConf
import wandb

import torch
from torch.utils.data import DataLoader

from models import models
from data.datasets import BrainMRInterSubj3D
from utils.image_io import save_nifti
from utils.misc import setup_dir

import random
random.seed(7)


def get_inference_dataloader(cfg, pin_memory=False):
    if cfg.data.name == 'brain_camcan':
        dataset = BrainMRInterSubj3D(data_dir_path=cfg.data.data_dir_path,
                                     crop_size=cfg.data.crop_size,
                                     modality=cfg.data.modality,
                                     atlas_path=cfg.data.atlas_path,
                                     voxel_size=cfg.data.voxel_size,
                                     evaluate=True)
    else:
        raise ValueError(f'Dataset config ({cfg.data}) not recognised.')
    return DataLoader(dataset,
                      shuffle=False,
                      pin_memory=pin_memory,
                      batch_size=cfg.data.batch_size,
                      num_workers=cfg.data.num_workers)


def set_up_kwargs(cfg):
    kwargs = {}
    kwargs["epochs"] = cfg.training.epochs
    kwargs["affine_epochs"] = cfg.training.affine_epochs
    kwargs["batch_size"] = cfg.data.batch_size
    kwargs["coords_batch_size"] = cfg.data.coords_batch_size
    kwargs["verbose"] = cfg.verbose

    if cfg.regularization.type == 'bending':
        kwargs["bending_regularization"] = True
        kwargs["hyper_regularization"] = False
        kwargs["jacobian_regularization"] = False
        kwargs["alpha_bending"] = cfg.regularization.alpha_bending

    kwargs["registration_type"] = cfg.network.type
    kwargs["network_type"] = cfg.network.activation
    kwargs["factor"] = cfg.network.factor
    kwargs["scale"] = cfg.network.scale
    kwargs["mapping_size"] = cfg.network.mapping_size
    kwargs["positional_encoding"] = cfg.network.positional_encoding

    kwargs["log_interval"] = cfg.training.log_interval
    kwargs["deformable_layers"] = cfg.network.deformable_layers
    kwargs["def_lr"] = cfg.training.def_lr
    kwargs["omega"] = cfg.network.omega
    kwargs["save_folder"] = setup_dir(os.getcwd() + '/logs')
    kwargs["loss_function"] = cfg.loss.type
    kwargs["use_mask"] = cfg.data.use_mask
    kwargs["image_shape"] = cfg.data.crop_size
    kwargs["voxel_size"] = cfg.data.voxel_size
    kwargs["transformation_type"] = cfg.transformation.type
    kwargs["cps"] = cfg.transformation.config.cps

    wandb.config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    return kwargs


def inr_inference(cfg, dataloader=None, out_dir='', device=torch.device('cpu'), save=False):

    # setup arguments for the model
    kwargs = set_up_kwargs(cfg)

    for idx, batch in enumerate(dataloader):
        print(f'\nSubject: {idx+1}/{len(dataloader)} \n')
        for k, x in batch.items():
            # reshape data for inference
            # 3d: (N=1, 1, H, W, D) -> (1, N=1, H, W, D)
            batch[k] = x.to(device=device).squeeze()

        ImpReg = models.ImplicitRegistrator(batch, idx, **kwargs)
        ImpReg.fit()

        # save the outputs
        subj_id = dataloader.dataset.subject_list[idx]
        output_id_dir = setup_dir(out_dir + f'/{subj_id}')
        for k, x in ImpReg.batch.items():
            x = x.detach().cpu().numpy()
            x = x.squeeze()
            save_nifti(x, path=output_id_dir + f'/{k}.nii.gz')

        torch.save(ImpReg.deform_net, f'{kwargs["save_folder"]}/{subj_id}_deform_net.pl')




@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    os.environ['WANDB_DISABLED'] = cfg.wandb_disable

    # configure GPU
    gpu = cfg.gpu
    if gpu is not None and isinstance(gpu, int):
        if not cfg.slurm:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # configure dataset & model
    dataloader = get_inference_dataloader(cfg, pin_memory=(device is torch.device('cuda')))

    # run inference
    if not cfg.analyse_only:
        output_dir = setup_dir(os.getcwd() + '/outputs')  # cwd = hydra.run.dir
        inr_inference(cfg, dataloader, output_dir, device=device)
    else:
        # TODO: add path
        output_dir = ''



if __name__ == '__main__':
    main()
