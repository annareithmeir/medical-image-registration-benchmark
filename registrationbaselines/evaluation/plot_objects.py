from pathlib import Path
import os

from typing import Tuple, Optional

import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import torch

from registrationbaselines.core.types import floatArray2D
from registrationbaselines.warping import utils_displacement
from registrationbaselines.evaluation import utils_evaluation
from registrationbaselines.metrics import metrics

os.environ['NEURITE_BACKEND'] = "pytorch"
matplotlib.rcParams['text.usetex'] = True


def plot_tensor_slices_difference(
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        save_path: str) -> None:
    """
    This function takes in two 3D tensors and creates a plot with 3 subplots (one row).
    Each subplot contains the difference of the corresponding middle slices.

    Parameters:
    tensor1 (torch.Tensor): The first 3D tensor.
    tensor2 (torch.Tensor): The second 3D tensor.
    """
    if tensor_a.shape != tensor_b.shape:
        raise ValueError("The two tensors must have the same shape.")

    # Calculate the middle index along the first dimension
    middle_index_0 = tensor_a.shape[0] // 2
    middle_index_1 = tensor_a.shape[1] // 2
    middle_index_2 = tensor_a.shape[2] // 2

    # Extract the middle slices from both tensors
    slice_a_0 = tensor_a[middle_index_0, :, :]
    slice_a_1 = tensor_a[:, middle_index_1, :]
    slice_a_2 = tensor_a[:, :, middle_index_2]

    slice_b_0 = tensor_b[middle_index_0, :, :]
    slice_b_1 = tensor_b[:, middle_index_1, :]
    slice_b_2 = tensor_b[:, :, middle_index_2]

    # Calculate the difference
    diff_0 = slice_a_0 - slice_b_0
    diff_1 = slice_a_1 - slice_b_1
    diff_2 = slice_a_2 - slice_b_2

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot the first middle slice
    axes[0].imshow(diff_0.numpy(), cmap='gray', vmin=-1, vmax=1)
    axes[0].set_title("Middle slice along axis 0")\

    # Plot the second middle slice
    axes[1].imshow(diff_1.numpy(), cmap='gray', vmin=-1, vmax=1)
    axes[1].set_title("Middle slice along axis 1")

    # Plot the third middle slice
    axes[2].imshow(diff_2.numpy(), cmap='gray', vmin=-1, vmax=1)
    axes[2].set_title("Middle slice along axis 2")

    # Display the plot
    fig.savefig(save_path)
    plt.close(fig)


def plot_quantitative_results(df: pd.DataFrame, plot_path: Path):
    df = df.drop(["min", "max", "mean", "std"])
    df = df.astype(float)

    # box plots
    df.plot(kind='box', subplots=True, figsize=(40, 7))
    plt.savefig(plot_path)


def plot_deformation_field(ax: plt.Axes, disp: np.ndarray, background: Optional[np.ndarray] = None,
                           interval: Optional[int] = 3, title: Optional[str] = None,
                           color: Optional[str] = 'cornflowerblue') -> None:
    """
    Plots 2d warped grid from a displacement field to a given matplotlib axis. source: https://github.com/qiuhuaqi/midir

    Bugfix: this now uses the pull convention to correctly display the deformation.

    :param ax: axis of a plt plot
    :param disp: displacement field of size (2,H,W)
    :param background:
    :param interval:
    :param background:  background image plotted behind the warped grid
    :param color: color of grid lines
    :return: None
    """
    if background is None:
        background = np.zeros(disp.shape[1:])

    assert disp.shape[0] == 2, "Displacement field should have shape (2, H, W)"

    # convert displacement from unit
    if disp.min() >= -1.0 and disp.max() <= 1.0:
        disp[0, ...] = float(disp.shape[1] - 1) * disp[0, ...] / 2.0
        disp[1, ...] = float(disp.shape[2] - 1) * disp[1, ...] / 2.0

    H, W = background.shape
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    # Create sampling grid
    sample_y = y[::interval, ::interval] - disp[0, ::interval, ::interval]
    sample_x = x[::interval, ::interval] - disp[1, ::interval, ::interval]

    # Plot deformed grid
    ax.plot(sample_x, sample_y, color=color, linewidth=0.34)
    ax.plot(sample_x.T, sample_y.T, color=color, linewidth=0.34)

    ax.set_title(title)
    ax.imshow(background, cmap='gray')
    # ax.set_aspect(1.25/1.75)
    ax.grid(False)
    ax.margins(x=0, y=0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)


def plot_all_registration_results(moving_image: torch.Tensor,
                                  fixed_image: torch.Tensor,
                                  pred_image: torch.Tensor,
                                  displacement: torch.Tensor,
                                  fixed_segmentations: Optional[torch.Tensor] = None,
                                  pred_segmentations: Optional[torch.Tensor] = None,
                                  moving_keypoints: Optional[torch.Tensor] = None,
                                  fixed_keypoints: Optional[torch.Tensor] = None,
                                  pred_keypoints: Optional[torch.Tensor] = None,
                                  title: Optional[str] = None,
                                  save_path: Path = None) -> plt.Figure:
    """
    plots a figure with 9x3 subplots. Half-slices used for plots in each dimension.
    rows: dims
    cols: M,F, warpedM,phi_grid, phi_quiver, diff_before, diff_after, labels, jacdet

    @param moving_image:
    @param fixed_image:
    @param pred_image:
    @param displacement: (h,w,d,3) unit-displacement
    @param fixed_segmentations:
    @param pred_segmentations:
    @param moving_keypoints:
    @param fixed_keypoints:
    @param pred_keypoints:
    @param title:
    @return: plot
    """

    assert utils_displacement.is_unit_displacement(displacement)

    moving_image = moving_image.numpy().squeeze()
    fixed_image = fixed_image.numpy().squeeze()
    pred_image = pred_image.numpy().squeeze()
    if fixed_segmentations is not None:
        fixed_segmentations = fixed_segmentations.numpy().squeeze()
    if pred_segmentations is not None:
        pred_segmentations = pred_segmentations.numpy().squeeze()
    if moving_keypoints is not None:
        moving_keypoints = moving_keypoints.numpy().squeeze()
    if fixed_keypoints is not None:
        fixed_keypoints = fixed_keypoints.numpy().squeeze()
    if pred_keypoints is not None:
        pred_keypoints = pred_keypoints.numpy().squeeze()

    displacement_denorm = utils_displacement.unit_displacement_to_displacement(
        displacement.detach().clone()).numpy().squeeze()
    displacement = displacement.numpy().squeeze()

    assert displacement.ndim in [
        3, 4], "Displacement field should have shape (h, w, d, 3) or (h, w, d, 3)"
    assert displacement.shape[-1] == 3 or displacement.shape[-1] == 2

    if displacement.shape[-1] != 3 and displacement.shape[0] == 3:
        displacement = displacement.transpose(1, 2, 3, 0)

    elif displacement.shape[-1] != 2 and displacement.shape[0] == 2:
        displacement = displacement.transpose(1, 2, 0)

    jacobian_determinant = metrics.jacobian_determinant_from_displacement(
        -displacement_denorm)  # we are pull convention, sitk function is push

    # BUGFIX: we need to swap the vector components to be (z,y,x) since this is torch convention. (x,y,z, [x,y,z]) -> (x,y,z, [z,y,x])
    displacement = displacement[..., [2, 1, 0]]

    fig = plt.figure(figsize=(40, 7))
    if title:
        fig.suptitle(title)
    image_size = moving_image.shape
    image_dim = displacement.shape[-1]

    # jacobian_determinant = metrics.jacobian_determinant_from_displacement(
    #     displacement_denorm)

    if (fixed_segmentations is not None) and (pred_segmentations is not None):
        num_cols = 9
    else:
        num_cols = 8

    if image_dim == 3:
        half_slice_idx = [int(s / 2) for s in image_size]
        toprow = True
        for d in np.arange(image_dim):
            dim_ls = [0, 1, 2]
            dim_ls.remove(d)

            # moving image
            ax = fig.add_subplot(3, num_cols, (num_cols * d) + 1)
            ax.imshow(moving_image.take(
                half_slice_idx[d], axis=d), cmap='gray')
            if moving_keypoints is not None:
                kp_slice = moving_keypoints[np.where(
                    abs(moving_keypoints[:, d] - half_slice_idx[d]) <= 0.5)]
                kp_slice = kp_slice[:, dim_ls]
                ax.scatter(kp_slice[:, 1], kp_slice[:, 0], marker='.', c='red')
            if toprow:
                ax.title.set_text("M")
            plt.axis('off')

            # fixed image
            ax = fig.add_subplot(3, num_cols, (num_cols * d) + 2)
            ax.imshow(fixed_image.take(half_slice_idx[d], axis=d), cmap='gray')
            if fixed_keypoints is not None:
                kp_slice = fixed_keypoints[np.where(
                    abs(fixed_keypoints[:, d] - half_slice_idx[d]) <= 0.5)]
                kp_slice = kp_slice[:, dim_ls]
                ax.scatter(kp_slice[:, 1], kp_slice[:, 0], marker='.', c='red')
            if toprow:
                ax.title.set_text("F")
            plt.axis('off')

            # deformed image
            ax = fig.add_subplot(3, num_cols, (num_cols * d) + 3)
            ax.imshow(pred_image.take(half_slice_idx[d], axis=d), cmap='gray')
            if pred_keypoints is not None:
                kp_slice = pred_keypoints[np.where(
                    abs(pred_keypoints[:, d] - half_slice_idx[d]) <= 0.5)]
                kp_slice = kp_slice[:, dim_ls]
                ax.scatter(kp_slice[:, 1], kp_slice[:, 0], marker='.', c='red')
            if toprow:
                ax.title.set_text("warped M")
            plt.axis('off')

            # displacement field
            ax = fig.add_subplot(3, num_cols, (num_cols * d) + 4)
            axes = [0, 1, 2]
            axes.remove(d)
            # print(displacement.shape, displacement[..., axes].take(half_slice_idx[d], axis=d).transpose(2, 0, 1).shape, pred_image.take(half_slice_idx[d], axis=d).shape)
            fieldAx = displacement[..., axes].take(half_slice_idx[d], axis=d)
            # plot_quiverplot(fieldAx, ax=ax)
            plot_deformation_field(ax, 1 * fieldAx.transpose(2, 0, 1), pred_image.take(
                half_slice_idx[d], axis=d), interval=8, color="white")
            ax.set_frame_on(False)
            if toprow:
                ax.title.set_text("deformation")
            plt.axis('off')
            # fig.tight_layout()

            # difference image before registration
            ax = fig.add_subplot(3, num_cols, (num_cols * d) + 5)
            diff_image = fixed_image.take(
                half_slice_idx[d], axis=d) - moving_image.take(half_slice_idx[d], axis=d)
            ax.imshow(diff_image, cmap='gray', vmin=-1, vmax=1)
            ax.set_frame_on(False)
            if toprow:
                ax.title.set_text("diff image")
            plt.axis('off')
            # fig.tight_layout()

            # difference image after registration
            ax = fig.add_subplot(3, num_cols, (num_cols * d) + 6)
            diff_image = fixed_image.take(
                half_slice_idx[d], axis=d) - pred_image.take(half_slice_idx[d], axis=d)
            ax.imshow(diff_image, cmap='gray', vmin=-1, vmax=1)
            ax.set_frame_on(False)
            if toprow:
                ax.title.set_text("diff image after")
            plt.axis('off')

            # jacobian determinant, negative values shown in red
            ax = fig.add_subplot(3, num_cols, (num_cols * d) + 7)
            jacdet_d = jacobian_determinant.take(half_slice_idx[d], axis=d)
            # jacdet_d[jacdet_d < 0] = np.min(jacdet_d)
            tmp = np.max(np.array([np.max(jacdet_d), -np.min(jacdet_d)]))
            norm = colors.TwoSlopeNorm(
                vmin=-tmp, vmax=tmp, vcenter=0)
            im1 = ax.imshow(jacdet_d, cmap="RdBu", norm=norm)
            ax.set_frame_on(False)
            if toprow:
                ax.title.set_text("jac det")
            plt.axis('off')

            # Add a colorbar with adjusted size
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im1, cax=cax)
            # Adjust the colorbar tick label size if needed
            cbar.ax.tick_params(labelsize=6)

            # discrete jacobian determinants
            ax = fig.add_subplot(3, num_cols, (num_cols * d) + 8)
            im1 = plot_jacobian_discrete_map(jacdet_d, 0.05, ax)
            ax.set_frame_on(False)
            if toprow:
                ax.title.set_text(
                    "jac det (folding:red, shrink:orange, vp:black, exp:blue)")
            plt.axis('off')

            # boundaries
            if (fixed_segmentations is not None) and (pred_segmentations is not None):
                ax = fig.add_subplot(3, num_cols, (num_cols * d) + 9)
                fixed_boundary = utils_evaluation.multilabel_to_boundary(
                    fixed_segmentations)
                pred_boundary = utils_evaluation.multilabel_to_boundary(
                    pred_segmentations)
                fixed_boundary = fixed_boundary.astype(np.int8)
                pred_boundary = pred_boundary.astype(np.int8)
                fixed_boundary[fixed_boundary > 0] = 1
                pred_boundary[pred_boundary > 0] = 1

                boundaries = fixed_boundary  # true=red
                boundaries[pred_boundary == 1] = 2  # pred=blue
                boundaries_slice = boundaries.take(half_slice_idx[d], axis=d)

                # cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=3)
                cmap = colors.ListedColormap(
                    ['w', 'crimson', 'cornflowerblue'])
                ax.imshow(boundaries_slice, cmap=cmap, interpolation='none')
                ax.set_frame_on(False)
                plt.axis('off')
                if toprow:
                    ax.title.set_text(
                        "segmentations (fixed: red, warped: blue)")
            toprow = False

    elif image_dim == 2:
        toprow = True

        # moving image
        ax = fig.add_subplot(3, num_cols, 1)
        ax.imshow(moving_image.squeeze(), cmap='gray')
        if moving_keypoints is not None:
            ax.scatter(
                moving_keypoints[:, 0], moving_keypoints[:, 1], marker='.', c='red')
        if toprow:
            ax.title.set_text("M")
        plt.axis('off')

        # fixed image
        ax = fig.add_subplot(3, num_cols, 2)
        ax.imshow(fixed_image.squeeze(), cmap='gray')
        if fixed_keypoints is not None:
            ax.scatter(fixed_keypoints[:, 0],
                       fixed_keypoints[:, 1], marker='.', c='red')
        if toprow:
            ax.title.set_text("F")
        plt.axis('off')

        # deformed image
        ax = fig.add_subplot(3, num_cols, 3)
        ax.imshow(pred_image.squeeze(), cmap='gray')
        if pred_keypoints is not None:
            ax.scatter(pred_keypoints[:, 0],
                       pred_keypoints[:, 1], marker='.', c='red')
        if toprow:
            ax.title.set_text("warped M")
        plt.axis('off')

        # displacement field
        ax = fig.add_subplot(3, num_cols, 4)
        fieldAx = displacement.squeeze()
        # plot_quiverplot(fieldAx, ax=ax)
        # neurite.plot.flow([displacement], show=False)
        plot_deformation_field(
            ax, 1 * fieldAx.transpose(2, 0, 1), pred_image.squeeze(), interval=5, color="white")
        ax.set_frame_on(False)
        if toprow:
            ax.title.set_text("deformation")
        plt.axis('off')
        fig.tight_layout()

        # difference image before registration
        ax = fig.add_subplot(3, num_cols, 5)
        diff_image = fixed_image.squeeze() - moving_image.squeeze()
        ax.imshow(diff_image, cmap='gray')
        ax.set_frame_on(False)
        if toprow:
            ax.title.set_text("diff image")
        plt.axis('off')
        # fig.tight_layout()

        # difference image after registration
        ax = fig.add_subplot(3, num_cols, 6)
        diff_image = fixed_image.squeeze() - pred_image.squeeze()
        ax.imshow(diff_image, cmap='gray')
        ax.set_frame_on(False)
        if toprow:
            ax.title.set_text("diff image after")
        plt.axis('off')

        # jacobian determinant, negative values shown in red
        ax = fig.add_subplot(3, num_cols,  8)
        jacdet_d = jacobian_determinant
        jacdet_d[jacdet_d < 0] = np.min(jacdet_d)
        norm = colors.TwoSlopeNorm(
            vmin=-np.max(jacdet_d), vmax=np.max(jacdet_d), vcenter=0)
        im1 = ax.imshow(jacdet_d, cmap="RdBu", norm=norm)
        ax.set_frame_on(False)
        if toprow:
            ax.title.set_text("jac det")
        plt.axis('off')
        plt.colorbar(im1, ax=ax)

        # boundaries
        ax = fig.add_subplot(3, num_cols,  7)
        if (fixed_segmentations is not None) and (pred_segmentations is not None):
            fixed_segmentations = fixed_segmentations.squeeze()
            pred_segmentations = pred_segmentations.squeeze()

            fixed_boundary = utils_evaluation.multilabel_to_boundary(
                fixed_segmentations)
            pred_boundary = utils_evaluation.multilabel_to_boundary(
                pred_segmentations)
            fixed_boundary = fixed_boundary.astype(np.int8)
            pred_boundary = pred_boundary.astype(np.int8)
            fixed_boundary[fixed_boundary > 0] = 1
            pred_boundary[pred_boundary > 0] = 1

            boundaries = fixed_boundary  # true=red
            boundaries[pred_boundary == 1] = 2  # pred=blue

            # cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=3)
            cmap = colors.ListedColormap(['w', 'crimson', 'cornflowerblue'])
            ax.imshow(boundaries, cmap=cmap, interpolation='none')
            ax.set_frame_on(False)
            ax.title.set_text("diff image after")
            plt.axis('off')

            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='crimson', edgecolor='black', label='True'),
                               Patch(facecolor='cornflowerblue', edgecolor='black', label='Predicted')]
            ax.legend(handles=legend_elements, loc='upper right')
        if toprow:
            ax.title.set_text("segmentations")

    else:
        print("Not implemented")

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.01, hspace=0.01)

    if save_path is not None:
        fig.savefig(save_path)
        plt.close(fig)
    else:
        fig.show()
    return fig


def plot_all_registration_results_debugging_wandb(moving_image: np.ndarray,
                                                  fixed_image: np.ndarray,
                                                  pred_image: np.ndarray,
                                                  displacement: np.ndarray,
                                                  fixed_labels: Optional[np.array] = None,
                                                  pred_labels: Optional[np.array] = None,
                                                  moving_keypoints: Optional[np.array] = None,
                                                  fixed_keypoints: Optional[np.array] = None,
                                                  pred_keypoints: Optional[np.array] = None,
                                                  title: Optional[str] = None):
    # 9x3 subplots, warp in shape (x,y,z,3)
    # rows: dims
    # cols: M,F, diff_before, warpedM,phi_grid, phi_quiver, diffimg,jacdet, jacdet_violin,maxshear
    assert displacement.shape[-1] == 3 or displacement.shape[-1] == 2

    # log_file = log_dir + '/debugging_results_epoch_' + str(step) + '.pdf'
    fig = plot_all_registration_results(None, moving_image, fixed_image, pred_image, displacement,
                                        fixed_labels=fixed_labels, pred_labels=pred_labels,
                                        moving_keypoints=moving_keypoints, fixed_keypoints=fixed_keypoints,
                                        pred_keypoints=pred_keypoints, title=title)
    return wandb.Image(fig)


def create_discrete_jacobian_det_map(jacdet: floatArray2D, tolerance: float) -> np.ndarray[Tuple[int, int],
                                                                                           np.dtype[np.int8]]:

    discrete_map = np.zeros_like(jacdet, dtype=np.int8)

    # folding
    discrete_map[jacdet < 0] = -1

    # shrinking
    shrinking_mask = (jacdet >= 0) & (jacdet < 1 - tolerance)
    discrete_map[shrinking_mask] = 0

    # constant volume
    constant_volume_mask = (
        jacdet >= 1 - tolerance) & (jacdet <= 1 + tolerance)
    discrete_map[constant_volume_mask] = 1

    # expansion
    discrete_map[jacdet > 1 + tolerance] = 2

    return discrete_map


def plot_jacobian_discrete_map(jacdet: floatArray2D, tolerance: float, ax: plt.Axes):
    discrete_jacobian_det = create_discrete_jacobian_det_map(jacdet, tolerance)

    # Find unique values in the discrete map
    unique_values = np.unique(discrete_jacobian_det)

    # Define the colors corresponding to the possible unique values
    value_to_color = {
        -1: 'red',     # folding
        0: 'orange',   # shrinking
        1: 'black',    # constant volume
        2: 'blue'      # expansion
    }

    # Create a colormap and norm based on the unique values
    colors_list = [value_to_color[val] for val in unique_values]
    cmap = colors.ListedColormap(colors_list)
    bounds = np.arange(min(unique_values)-0.5, max(unique_values)+1.5)
    norm = colors.BoundaryNorm(bounds, cmap.N)

    # Plot the map
    im1 = ax.imshow(discrete_jacobian_det, cmap=cmap, norm=norm)
    return im1


def plot_max_shear(ax: plt.Axes, displacement_field: np.ndarray) -> None:
    pass


def plot_quiverplot(u: np.ndarray, axis: Optional[int] = None, ax=None) -> None:
    """
    :param u: (h,d,w,3)
    """

    def flow(slices_in,  # the 2D slices
             ax,
             titles=None,  # list of titles
             cmaps=None,  # list of colormaps
             width=15,  # width in in
             indexing='ij',  # plot vecs w/ matrix indexing 'ij' or cartesian indexing 'xy'
             img_indexing=True,  # whether to match the image view, i.e. flip y axis
             grid=False,  # option to plot the images in a grid or a single row
             show=True,  # option to actually show the plot (plt.show())
             quiver_width=None,
             plot_block=True,  # option to plt.show()
             scale=1):  # note quiver essentially draws quiver length = 1/scale
        '''
        plot a grid of flows (2d+2 images)
        '''

        # input processing
        nb_plots = len(slices_in)
        for slice_in in slices_in:
            assert len(
                slice_in.shape) == 3, 'each slice has to be 3d: 2d+2 channels'
            assert slice_in.shape[-1] == 2, 'each slice has to be 3d: 2d+2 channels'

        def input_check(inputs, nb_plots, name):
            ''' change input from None/single-link '''
            if not isinstance(inputs, (list, tuple)):
                inputs = [inputs]
            assert (inputs is None) or (len(inputs) == nb_plots) or (len(inputs) == 1), \
                'number of %s is incorrect' % name
            if inputs is None:
                inputs = [None]
            if len(inputs) == 1:
                inputs = [inputs[0] for i in range(nb_plots)]
            return inputs

        assert indexing in ['ij', 'xy']
        # Since img_indexing, indexing may modify slices_in in memory
        slices_in = np.copy(slices_in)

        if indexing == 'ij':
            for si, slc in enumerate(slices_in):
                # Make y values negative so y-axis will point down in plot
                slices_in[si][:, :, 1] = -slices_in[si][:, :, 1]

        if img_indexing:
            for si, slc in enumerate(slices_in):
                # Flip vertical order of y values
                slices_in[si] = np.flipud(slc)

        titles = input_check(titles, nb_plots, 'titles')
        cmaps = input_check(cmaps, nb_plots, 'cmaps')
        scale = input_check(scale, nb_plots, 'scale')

        # figure out the number of rows and columns
        if grid:
            if isinstance(grid, bool):
                rows = np.floor(np.sqrt(nb_plots)).astype(int)
                cols = np.ceil(nb_plots / rows).astype(int)
            else:
                assert isinstance(grid, (list, tuple)), \
                    "grid should either be bool or [rows,cols]"
                rows, cols = grid
        else:
            rows = 1
            cols = nb_plots

        # # prepare the subplot
        # fig, axs = plt.subplots(rows, cols)
        # if rows == 1 and cols == 1:
        #     axs = [axs]

        for i in range(nb_plots):
            col = np.remainder(i, cols)
            row = np.floor(i / cols).astype(int)

            # get row and column axes
            # row_axs = axs if rows == 1 else axs[row]
            # ax = row_axs[col]

            # turn off axis
            ax.axis('off')

            # add titles
            if titles is not None and titles[i] is not None:
                ax.title.set_text(titles[i])

            u, v = slices_in[i][..., 0], slices_in[i][..., 1]
            colors = np.arctan2(u, v)
            colors[np.isnan(colors)] = 0
            norm = colors.Normalize()
            norm.autoscale(colors)
            if cmaps[i] is None:
                colormap = cm.winter
            else:
                raise Exception(
                    "custom cmaps not currently implemented for plt.flow()")

            # show figure
            colormap = cm.hsv
            step = 10
            # X = x[::step, ::step]
            # Y = y[::step, ::step]
            u = u[::step, ::step]
            v = v[::step, ::step]
            ax.quiver(u, v,
                      color=colormap(norm(colors).flatten()),
                      angles='xy',
                      units='xy',
                      width=quiver_width,
                      scale=scale[i])
            ax.axis('equal')

        # clear axes that are unnecessary
        # for i in range(nb_plots, col * row):
        #     col = np.remainder(i, cols)
        #     row = np.floor(i / cols).astype(int)
        #
        #     # get row and column axes
        #     row_axs = axs if rows == 1 else axs[row]
        #     ax = row_axs[col]
        #
        #     ax.axis('off')

        # show the plots
        # fig.set_size_inches(width, rows / cols * width)
        # plt.tight_layout()
        #
        # if show:
        #     plt.show(block=plot_block)
        #
        # return (fig, axs)

    if u.shape[-1] == 3:
        assert axis is not None
        axes = [0, 1, 2]
        axes.remove(axis)
        u = u[..., axes].take(u.shape[axis] // 2, axis=axis)
        # print(u.shape)

    flow([u], show=False, ax=ax, scale=3)
    # f, axs =neurite.plot.flow([u_slice], show=False, ax=ax)
