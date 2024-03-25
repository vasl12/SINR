import torch
import torch.nn as nn
import torch.optim as optim
import os
import tqdm
from utils import general
from networks import networks
from objectives import ncc
from objectives import regularizers
from utils.general import set_kernel
from objectives.nmi import NMI
import numpy as np
from deepali.core.bspline import evaluate_cubic_bspline
from deepali.spatial.bspline import FreeFormDeformation
from deepali.core import Grid
from utils.metric import measure_seg_metrics, measure_disp_metrics


class input_mapping(nn.Module):
    def __init__(self, B=None, factor=1.0):
        super(input_mapping, self).__init__()
        self.B = factor * B

    def forward(self, x):
        x_proj = (2. * np.pi * x) @ self.B.T
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class ImplicitRegistrator:
    """This is a class for registrating implicitly represented images."""

    def __call__(
        self, coordinate_tensor=None, output_shape=(28, 28), dimension=0, slice_pos=0
    ):
        """Return the image-values for the given input-coordinates."""

        # Use standard coordinate tensor if none is given
        if coordinate_tensor is None:
            coordinate_tensor = self.make_coordinate_slice(
                output_shape, dimension, slice_pos
            )

        output = self.network(coordinate_tensor)

        # Shift coordinates by 1/n * v
        coord_temp = torch.add(output, coordinate_tensor)

        transformed_image = self.transform_no_add(coord_temp)
        return (
            transformed_image.cpu()
            .detach()
            .numpy()
            .reshape(output_shape[0], output_shape[1])
        )

    def __init__(self, batch, batch_idx, **kwargs):
        """Initialize the learning model."""

        # Set all default arguments in a dict: self.args
        self.set_default_arguments()
        self.batch = batch
        self.batch_idx = batch_idx

        # Check if all kwargs keys are valid (this checks for typos)
        # assert all(kwarg in self.args.keys() for kwarg in kwargs)

        # Parse important argument from kwargs
        self.epochs = kwargs["epochs"] if "epochs" in kwargs else self.args["epochs"]
        self.affine_epochs = kwargs["affine_epochs"] if "affine_epochs" in kwargs else self.args["epochs"]
        self.log_interval = (
            kwargs["log_interval"]
            if "log_interval" in kwargs
            else self.args["log_interval"]
        )
        self.gpu = kwargs["gpu"] if "gpu" in kwargs else self.args["gpu"]
        self.def_lr = kwargs["def_lr"] if "def_lr" in kwargs else self.args["def_lr"]
        self.momentum = (
            kwargs["momentum"] if "momentum" in kwargs else self.args["momentum"]
        )
        self.optimizer_arg = (
            kwargs["optimizer"] if "optimizer" in kwargs else self.args["optimizer"]
        )
        self.loss_function_arg = (
            kwargs["loss_function"]
            if "loss_function" in kwargs
            else self.args["loss_function"]
        )
        self.weight_init = (
            kwargs["weight_init"]
            if "weight_init" in kwargs
            else self.args["weight_init"]
        )
        self.deformable_layers = kwargs["deformable_layers"] if "deformable_layers" in kwargs else self.args["deformable_layers"]
        self.weight_init = (
            kwargs["weight_init"]
            if "weight_init" in kwargs
            else self.args["weight_init"]
        )
        self.omega = kwargs["omega"] if "omega" in kwargs else self.args["omega"]
        self.save_folder = (
            kwargs["save_folder"]
            if "save_folder" in kwargs
            else self.args["save_folder"]
        )

        # Parse other arguments from kwargs
        self.verbose = (
            kwargs["verbose"] if "verbose" in kwargs else self.args["verbose"]
        )

        self.cps = (kwargs["cps"] if "cps" in kwargs else self.args["cps"])

        # Make folder for output
        if not self.save_folder == "" and not os.path.isdir(self.save_folder):
            os.mkdir(self.save_folder)

        # Add slash to divide folder and filename
        self.save_folder += "/"

        # Make loss list to save losses
        self.loss_list = [0 for _ in range(self.epochs)]
        self.data_loss_list = [0 for _ in range(self.epochs)]

        # Set seed
        torch.manual_seed(self.args["seed"])

        # Load network
        self.network_from_file = (
            kwargs["network"] if "network" in kwargs else self.args["network"]
        )
        self.network_type = (
            kwargs["network_type"]
            if "network_type" in kwargs
            else self.args["network_type"]
        )

        self.positional_encoding = (
            kwargs["positional_encoding"]
            if "positional_encoding" in kwargs
            else self.args["positional_encoding"]
        )

        self.registration_type = (
            kwargs["registration_type"]
            if "registration_type" in kwargs
            else self.args["registration_type"]
        )

        self.mapping_size = (
            kwargs["mapping_size"]
            if "mapping_size" in kwargs
            else self.args["mapping_size"]
        )
        self.scale = (
            kwargs["scale"]
            if "scale" in kwargs
            else self.args["scale"]
        )
        self.factor = (
            kwargs["factor"]
            if "factor" in kwargs
            else self.args["factor"]
        )

        self.log_interval = (
            kwargs["log_interval"]
            if "log_interval" in kwargs
            else self.args["log_interval"]
        )

        if self.network_from_file is None:
            if self.network_type == "MLP":
                self.deform_net = networks.MLP(self.deformable_layers)
            elif self.network_type == "Siren":
                self.deform_net = networks.Siren(self.deformable_layers, self.weight_init, self.omega)
            if self.verbose:
                if self.registration_type == "deformable":
                    print(
                        "Deformable network contains {} trainable parameters.".format(
                            general.count_parameters(self.deform_net)
                        )
                    )
        else:
            # TODO:
            self.network = torch.load(self.network_from_file)
            if self.gpu:
                self.network.cuda()


        # TODO: lr to aff_lr and def_lr
        # Choose the optimizer
        if self.optimizer_arg.lower() == "sgd":
            self.optimizer = optim.SGD(
                self.network.parameters(), lr=self.def_lr, momentum=self.momentum
            )
        elif self.optimizer_arg.lower() == "adam":
            self.optimizer = optim.Adam(params=[{"name": "Def", "params": self.deform_net.parameters(), "lr": self.def_lr}])
        elif self.optimizer_arg.lower() == "adadelta":
            self.optimizer = optim.Adadelta(self.network.parameters(), lr=self.def_lr)

        else:
            self.optimizer = optim.SGD(
                self.network.parameters(), lr=self.def_lr, momentum=self.momentum
            )
            print(
                "WARNING: "
                + str(self.optimizer_arg)
                + " not recognized as optimizer, picked SGD instead"
            )

        # Choose the loss function
        if self.loss_function_arg.lower() == "mse":
            self.criterion = nn.MSELoss()

        elif self.loss_function_arg.lower() == "l1":
            self.criterion = nn.L1Loss()

        elif self.loss_function_arg.lower() == "ncc":
            self.criterion = ncc.NCC()

        elif self.loss_function_arg.lower() == "nmi":
            self.criterion = NMI(sigma=0.1, nbins=64)

        elif self.loss_function_arg.lower() == "smoothl1":
            self.criterion = nn.SmoothL1Loss(beta=0.2)

        elif self.loss_function_arg.lower() == "huber":
            self.criterion = nn.HuberLoss()
        else:
            self.criterion = nn.MSELoss()
            print(
                "WARNING: "
                + str(self.loss_function_arg)
                + " not recognized as loss function, picked MSE instead"
            )

        # Move variables to GPU
        if self.gpu:
            self.deform_net.cuda()

        # Parse regularization kwargs
        self.jacobian_regularization = (
            kwargs["jacobian_regularization"]
            if "jacobian_regularization" in kwargs
            else self.args["jacobian_regularization"]
        )
        self.alpha_jacobian = (
            kwargs["alpha_jacobian"]
            if "alpha_jacobian" in kwargs
            else self.args["alpha_jacobian"]
        )

        self.hyper_regularization = (
            kwargs["hyper_regularization"]
            if "hyper_regularization" in kwargs
            else self.args["hyper_regularization"]
        )
        self.alpha_hyper = (
            kwargs["alpha_hyper"]
            if "alpha_hyper" in kwargs
            else self.args["alpha_hyper"]
        )

        self.bending_regularization = (
            kwargs["bending_regularization"]
            if "bending_regularization" in kwargs
            else self.args["bending_regularization"]
        )
        self.alpha_bending = (
            kwargs["alpha_bending"]
            if "alpha_bending" in kwargs
            else self.args["alpha_bending"]
        )

        # Parse arguments from kwargs
        self.image_shape = (
            kwargs["image_shape"]
            if "image_shape" in kwargs
            else self.args["image_shape"]
        )
        self.batch_size = (
            kwargs["batch_size"] if "batch_size" in kwargs else self.args["batch_size"]
        )
        self.coord_batch_size = (
            kwargs["coords_batch_size"] if "coords_batch_size" in kwargs else self.args["coords_batch_size"]
        )
        self.transformation_type = (
            kwargs["transformation_type"] if "transformation_type" in kwargs else self.args["transformation_type"]
        )

        # Initialization
        self.moving_image = batch['moving']
        self.fixed_image = batch['fixed']
        self.fixed_mask = batch['fixed_mask']
        self.moving_mask = batch['moving_mask']
        self.mask = torch.logical_or(batch['fixed_mask'], batch['moving_mask']).float()
        self.fixed_seg = batch['fixed_seg']
        self.moving_seg = batch['moving_seg']

        self.batch = {}

        if self.transformation_type == 'bspline':
            k = set_kernel([self.cps, self.cps, self.cps])
            self.padding = [(len(k) - 1) // 2 for k in k]

            grid = Grid(self.fixed_image.shape)
            self.kernel = FreeFormDeformation(grid, stride=self.cps).to("cuda:0").kernel()
            self.transformation = evaluate_cubic_bspline
            self.padding = 2*self.padding
            self.padded_fixed = torch.nn.functional.pad(self.fixed_image, self.padding, value=0)
            self.padded_mask = torch.nn.functional.pad(self.fixed_mask*self.moving_mask, self.padding, value=0)

            # TODO: we have to account for non cubic images
            self.num_pixels = torch.prod(torch.tensor(self.padded_fixed.shape))
            indices_x = torch.arange(self.cps//2, self.padded_fixed.shape[0], step=self.cps)#[1:]
            indices_y = torch.arange(self.cps//2, self.padded_fixed.shape[1], step=self.cps)
            indices_z = torch.arange(self.cps//2, self.padded_fixed.shape[2], step=self.cps)
            self.sh1 = len(indices_x)
            self.sh2 = len(indices_y)
            self.sh3 = len(indices_z)
            x, y, z = torch.meshgrid(indices_x, indices_y, indices_z, indexing='ij')
            indices_z = (z + y * self.padded_fixed.shape[0] + x * self.padded_fixed.shape[1] * self.padded_fixed.shape[2]).flatten()
            self.control_point_indices = indices_z

            self.possible_coordinate_tensor = general.make_coordinate_tensor(self.padded_fixed.shape)
            self.control_points = self.possible_coordinate_tensor[self.control_point_indices]
            control_point_mask = self.padded_mask.flatten()[self.control_point_indices]
            self.control_points_masked = self.control_points[control_point_mask > 0]

        elif self.transformation_type == 'dense':
            self.sh = self.fixed_image.shape[0]
            # self.possible_coordinate_tensor = general.make_masked_coordinate_tensor(self.mask.cpu(), self.fixed_image.shape)
            self.possible_coordinate_tensor = general.make_coordinate_tensor(self.fixed_image.shape)
            self.possible_coordinate_tensor_val = general.make_coordinate_tensor(self.fixed_image.shape)


        if self.gpu:
            self.moving_image = self.moving_image.cuda()
            self.fixed_image = self.fixed_image.cuda()
            # self.padded_fixed = self.padded_fixed.cuda()
            self.moving_mask = self.moving_mask.cuda()
            self.fixed_mask = self.fixed_mask.cuda()
            self.moving_seg = self.moving_seg.cuda()
            self.fixed_seg = self.fixed_seg.cuda()

    def set_default_arguments(self):
        """Set default arguments."""

        # Inherit default arguments from standard learning model
        self.args = {}

        # Define the value of arguments
        self.args["mask"] = None
        self.args["mask_2"] = None
        self.args["method"] = 1

        self.args["lr"] = 0.00001
        self.args["aff_lr"] = 0.00001
        self.args["def_lr"] = 0.00001
        self.args["coords_batch_size"] = 1000
        self.args["deformable_layers"] = [3, 256, 256, 256, 3]
        self.args["velocity_steps"] = 1

        self.args["cps"] = 1

        # Define argument defaults specific to this class
        self.args["output_regularization"] = False
        self.args["alpha_output"] = 0.2
        self.args["reg_norm_output"] = 1

        self.args["jacobian_regularization"] = False
        self.args["alpha_jacobian"] = 0.05

        self.args["hyper_regularization"] = False
        self.args["alpha_hyper"] = 0.25

        self.args["bending_regularization"] = False
        self.args["alpha_bending"] = 10.0

        self.args["image_shape"] = (200, 200)
        self.args["network"] = None

        self.args["epochs"] = 2500
        self.args["log_interval"] = self.args["epochs"] // 4
        self.args["verbose"] = True
        self.args["save_folder"] = "output"

        self.args["network_type"] = "Siren"
        self.args["registration_type"] = "deformable"

        self.args["gpu"] = torch.cuda.is_available()
        self.args["optimizer"] = "Adam"
        self.args["loss_function"] = "ncc"
        self.args["momentum"] = 0.5

        self.args["positional_encoding"] = False
        self.args["weight_init"] = True
        self.args["omega"] = 32
        self.args["seed"] = 1

        self.transformation_type = 'dense'
        self.mapping_size = 256
        self.scale = 4.0
        self.factor = 1.0
        self.log_interval = 500


    def training_iteration(self, epoch, flag=100, coordinate_tensor=None, model_input=None):
        """Perform one iteration of training."""

        if self.transformation_type == 'dense':
            # indices of the coordinates that should be processed in this batch
            indices = torch.randperm(
                self.possible_coordinate_tensor.shape[0], device="cuda"
            )[: self.coord_batch_size]

            model_input = model_input[indices, :]
            coordinate_tensor = coordinate_tensor[indices, :]
            coordinate_tensor.requires_grad = True
            model_input.requires_grad = True
            coordinate_tensor.retain_grad()

        self.deform_net.train()

        loss = 0
        aff_coords = coordinate_tensor
        output = self.deform_net.forward(model_input)
        def_coords = torch.add(output, aff_coords)

        image_size = self.fixed_image.shape

        if self.transformation_type == 'bspline':
            def_coords = def_coords.reshape(1, self.sh1, self.sh2, self.sh3, 3).permute(0, 4, 1, 2, 3)
            coordinate_tensor = coordinate_tensor.reshape(1, self.sh1, self.sh2, self.sh3, 3).permute(0, 4, 1, 2, 3)
            def_coords = self.transformation(def_coords, stride=self.cps, shape=image_size, kernel=self.kernel)
            coordinate_tensor = self.transformation(coordinate_tensor, stride=self.cps, shape=image_size, kernel=self.kernel)
            coordinate_tensor = coordinate_tensor.permute(0, 2, 3, 4, 1)
            def_coords = def_coords.permute(0, 2, 3, 4, 1)

            fixed_image = torch.nn.functional.grid_sample(self.fixed_image[None, None], coordinate_tensor,
                                                          align_corners=True).squeeze()
            transformed_image = torch.nn.functional.grid_sample(self.moving_image[None, None], def_coords,
                                                                align_corners=True).squeeze()
        else:

            fixed_image = torch.nn.functional.grid_sample(self.fixed_image[None, None], coordinate_tensor[None, None, None],
                                        align_corners=True).squeeze()

            transformed_image = torch.nn.functional.grid_sample(self.moving_image[None, None], def_coords[None, None, None],
                                                align_corners=True).squeeze()

        aff_coords = coordinate_tensor

        # Compute the loss
        loss += self.criterion(fixed_image[None, None], transformed_image[None, None])

        # Store the value of the data loss
        if self.verbose:
            self.data_loss_list[epoch] = loss.detach().cpu().numpy().item()

        output_rel = torch.subtract(def_coords, aff_coords)
        # Regularization
        if self.bending_regularization:
            if self.transformation_type == 'dense':
                bending_reg = regularizers.bending_energy_loss(output_rel[None][None].permute(0, 1, 3, 2))
            elif self.transformation_type == 'bspline':
                bending_reg = regularizers.l2reg_loss(output_rel.permute(0, 4, 1, 2, 3))

            loss += self.alpha_bending * bending_reg

        for param in self.deform_net.parameters():
            param.grad = None

        loss.backward(retain_graph=True)
        self.optimizer.step()

        # Store the value of the total loss
        if self.verbose:
            self.loss_list[epoch] = loss.detach().cpu().numpy()

    def validation_iteration(self, epoch, coordinate_tensor=None, model_input=None):

        self.deform_net.eval()
        loss = 0
        with (torch.no_grad()):
            aff_coords = coordinate_tensor
            output = self.deform_net.forward(model_input)
            def_coords = torch.add(output, aff_coords)

            image_size = self.fixed_image.shape

            if self.transformation_type == 'bspline':
                def_coords = def_coords.reshape(1, self.sh1, self.sh2, self.sh3, 3).permute(0, 4, 1, 2, 3)
                coordinate_tensor = coordinate_tensor.reshape(1, self.sh1, self.sh2, self.sh3, 3).permute(0, 4, 1, 2, 3)
                def_coords = self.transformation(def_coords, stride=self.cps, shape=image_size, kernel=self.kernel)
                coordinate_tensor = self.transformation(coordinate_tensor, stride=self.cps, shape=image_size, kernel=self.kernel)
                coordinate_tensor = coordinate_tensor.permute(0, 2, 3, 4, 1)
                def_coords = def_coords.permute(0, 2, 3, 4, 1)
            else:
                coordinate_tensor = coordinate_tensor.reshape(1, image_size[0], image_size[1], image_size[2], 3)
                def_coords = def_coords.reshape(1, image_size[0], image_size[1], image_size[2], 3)

            aff_coords = coordinate_tensor
            self.batch['disp_pred'] = def_coords - aff_coords

            fixed_image = torch.nn.functional.grid_sample(self.fixed_image[None, None], coordinate_tensor, align_corners=True).squeeze()

            self.batch['fixed'] = fixed_image
            moving_image = torch.nn.functional.grid_sample(self.moving_image[None, None], coordinate_tensor, align_corners=True).squeeze()

            self.batch['moving'] = moving_image
            fixed_seg = torch.nn.functional.grid_sample(self.fixed_seg[None, None], coordinate_tensor, align_corners=True, mode='nearest').squeeze()

            self.batch['fixed_seg'] = fixed_seg
            fixed_mask = torch.nn.functional.grid_sample(self.fixed_mask[None, None], coordinate_tensor, align_corners=True, mode='nearest').squeeze()

            self.batch['fixed_mask'] = fixed_mask
            moving_seg = torch.nn.functional.grid_sample(self.moving_seg[None, None], coordinate_tensor, align_corners=True, mode='nearest').squeeze()

            self.batch['moving_seg'] = moving_seg
            transformed_image = torch.nn.functional.grid_sample(self.moving_image[None, None], def_coords, align_corners=True).squeeze()

            self.batch['warped_moving'] = transformed_image
            warped_moving_seg = torch.nn.functional.grid_sample(self.moving_seg[None, None], def_coords, align_corners=True, mode='nearest').squeeze()

            self.batch['warped_seg'] = warped_moving_seg

            # Compute the loss
            loss += self.criterion(fixed_image[None, None], transformed_image[None, None])

            # Store the value of the data loss
            if self.verbose:
                self.data_loss_list[epoch] = loss.detach().cpu().numpy().item()

            # Relativation of output
            output_rel = torch.subtract(def_coords, aff_coords)
            # Regularization
            if self.bending_regularization:
                if self.transformation_type == 'dense':
                    bending_reg = regularizers.bending_energy_loss(output_rel.permute(0, 4, 1, 2, 3))
                elif self.transformation_type == 'bspline':
                    bending_reg = regularizers.l2reg_loss(output_rel.permute(0, 4, 1, 2, 3))
                loss += self.alpha_bending * bending_reg

            field = (def_coords-aff_coords).reshape(image_size[0]*image_size[1]*image_size[2], 3) #* self.fixed_mask.flatten().repeat(3, 1).permute(1, 0)
            general.visualise_results(fixed_image.clone().reshape(image_size[0], image_size[1], image_size[2]).detach(), moving_image.clone().detach(), transformed_image, field.clone().detach().cpu(), title='Validation intensities', show=False, cmap='gray')
            general.visualise_results(fixed_seg.clone().reshape(image_size[0], image_size[1], image_size[2]).detach(), moving_seg.clone().detach(), warped_moving_seg, field.clone().detach().cpu(), title='Validation segmentations', show=False, cmap='jet')


            for param in self.deform_net.parameters():
                param.grad = None

            metrics = {}
            seg_res = measure_seg_metrics({'fixed_seg': fixed_seg.clone().detach(), 'warped_seg': warped_moving_seg.clone().detach()})
            metrics.update({f'{k}': metric.item() for k, metric in seg_res.items()})

            disp_res = measure_disp_metrics({'disp_pred': self.batch['disp_pred'].cpu().permute(0, 4, 1, 2, 3)})
            metrics.update({f'{k}': metric.item() for k, metric in disp_res.items()})

            # stopping criteria
            flag = 0
            if disp_res['folding_ratio'] * 100 >= 0.9:
                print(f'\n1% folding ratio reached at epoch {epoch}\n')
                flag = 1

            return metrics, flag


    def transform_no_add(self, transformation, moving_image=None, reshape=False, interp='linear'):
        """Transform moving image given a transformation."""

        # If no moving image is given use the standard one
        if moving_image is None:
            moving_image = self.moving_image
        # print('GET MOVING')
        if interp == 'linear':
            transformed_image = general.fast_trilinear_interpolation(
                moving_image,
                transformation[:, 0],
                transformation[:, 1],
                transformation[:, 2],
            )
        elif interp == 'nearest':
            transformed_image = general.fast_nearest_interpolation(
                moving_image,
                transformation[:, 0],
                transformation[:, 1],
                transformation[:, 2],
            )
        return transformed_image


    def fit(self, epochs=None):
        """Train the network."""

        # Determine epochs
        if epochs is None:
            epochs = self.epochs

        # Set seed
        torch.manual_seed(self.args["seed"])

        # Extend lost_list if necessary
        if not len(self.loss_list) == epochs:
            self.loss_list = [0 for _ in range(epochs)]
            self.data_loss_list = [0 for _ in range(epochs)]

        if self.transformation_type == 'bspline':
            coordinate_tensor_train = self.control_points
            coordinate_tensor_val = self.control_points.clone()
            coordinate_tensor_train = coordinate_tensor_train.requires_grad_(True)
            coordinate_tensor_train.retain_grad()
        elif self.transformation_type == 'dense':
            coordinate_tensor_train = self.possible_coordinate_tensor
            coordinate_tensor_val = self.possible_coordinate_tensor_val


        if self.positional_encoding:
            mapping_size = self.mapping_size # of FF
            B_gauss = torch.tensor(np.random.normal(scale=self.scale, size=(mapping_size, 3)),
                                   dtype=torch.float32, device="cuda")
            input_mapper = input_mapping(B=B_gauss, factor=self.factor).to("cuda")
            model_input_train = input_mapper(coordinate_tensor_train)
            model_input_val = input_mapper(coordinate_tensor_val)
        else:
            model_input_train = coordinate_tensor_train
            model_input_val = coordinate_tensor_val

        # Perform training iterations
        for i in tqdm.tqdm(range(epochs-1)):
            self.training_iteration(i, self.affine_epochs, coordinate_tensor_train, model_input_train)
            if i % self.log_interval == 0:
                _, flag = self.validation_iteration(i, coordinate_tensor_val, model_input_val)
                if flag:
                    break

        print('\nTesting:\n')
        metrics, _ = self.validation_iteration(i, coordinate_tensor_val, model_input_val)

