import os
import nibabel as nib
from dipy.tracking import utils
from dipy.core.gradients import gradient_table
from dipy.tracking.local_tracking import LocalTracking
import numpy as np

# from FT.all_subj import all_subj_names, all_subj_folders


def round_seg(seg_file):
    img = nib.load(seg_file)
    orig_data = img.get_fdata()
    new_data = np.round(orig_data)
    new_img = nib.Nifti1Image(new_data.astype(int), img.affine)
    return new_img


def load_dwi_files(folder_name, small_delta=15.5):
    """
    param:
        folder_name -
        small_delta -
    return:
        white_matter - the entire brain mask to track fibers
    """
    dwi_folder = f"{folder_name}/dwi"
    for file in os.listdir(dwi_folder):
        if file.endswith(".bvec"):
            bvec_file = os.path.join(dwi_folder, file)
    labels_file_name = f"{dwi_folder}/tmp/Segmentation.nii.gz"
    labels_img = round_seg(labels_file_name)
    bval_file = bvec_file.replace("bvec", "bval")
    nii_file = os.path.join(dwi_folder, "diff_corrected.nii.gz")
    hardi_img = nib.load(nii_file)
    data = hardi_img.get_data()
    affine = hardi_img.affine
    gtab = gradient_table(bval_file, bvec_file, small_delta=small_delta)
    # labels_img = nib.load(labels_file_name)
    labels = labels_img.get_data()
    white_matter = labels == 3  # | (labels == 2)  # 3-WM, 2-GM

    return gtab, data, affine, labels, white_matter, nii_file, bvec_file


def load_mask(folder_name, mask_type):
    file_list = os.listdir(folder_name)
    for file in file_list:
        if "mask" in file and mask_type in file and file.endswith(".nii"):
            mask_file = os.path.join(folder_name, file)
            mask_img = nib.load(mask_file)
            mask_mat = mask_img.get_data()
    return mask_mat


def create_seeds(
    folder_name, lab_labels_index, affine, use_mask=True, mask_type="cc", den=1
):
    if use_mask:
        mask_mat = load_mask(folder_name, mask_type)
        seed_mask = mask_mat == 1
    else:
        seed_mask = lab_labels_index > 0  # GM seeds
    seeds = utils.seeds_from_mask(seed_mask, density=den, affine=affine)
    return seeds


def create_seeds_new(labels, affine, den=1):
    seed_mask = labels == 3
    seeds = utils.seeds_from_mask(seed_mask, density=den, affine=affine)
    return seeds


def create_csd_model(data, gtab, white_matter, sh_order=6):
    from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, auto_response

    response, ratio = auto_response(gtab, data, roi_radius=10, fa_thr=0.7)

    csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=sh_order)
    csd_fit = csd_model.fit(data, mask=white_matter)

    return csd_fit


def create_fa_classifier(gtab, data, white_matter):
    import dipy.reconst.dti as dti
    from dipy.reconst.dti import fractional_anisotropy
    from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion

    tensor_model = dti.TensorModel(gtab)
    tenfit = tensor_model.fit(data, mask=white_matter)
    fa = fractional_anisotropy(tenfit.evals)
    classifier = ThresholdStoppingCriterion(fa, 0.18)

    return fa, classifier


def create_act_classifier(fa, folder_name, labels):  # Does not work
    from dipy.tracking.stopping_criterion import ActStoppingCriterion

    background = np.ones(labels.shape)
    background[(np.asarray(labels) > 0) > 0] = 0
    include_map = np.zeros(fa.shape)
    lab = folder_name + r"\rMegaAtlas_cortex_Labels.nii"
    lab_file = nib.load(lab)
    lab_labels = lab_file.get_data()
    include_map[background > 0] = 1
    include_map[lab_labels > 0] = 1
    include_map[fa > 0.18] = 1
    include_map = include_map == 1
    exclude_map = labels == 1

    act_classifier = ActStoppingCriterion(include_map, exclude_map)

    return act_classifier


def create_streamlines(csd_fit, classifier, seeds, affine):
    from dipy.data import default_sphere
    from dipy.direction import DeterministicMaximumDirectionGetter
    from dipy.tracking.streamline import Streamlines

    detmax_dg = DeterministicMaximumDirectionGetter.from_shcoeff(
        csd_fit.shm_coeff, max_angle=30.0, sphere=default_sphere
    )

    streamlines = Streamlines(
        LocalTracking(
            detmax_dg, classifier, seeds, affine, step_size=1, return_all=False
        )
    )

    long_streamlines = np.ones((len(streamlines)), bool)
    for i in range(0, len(streamlines)):
        if streamlines[i].shape[0] < 100:
            long_streamlines[i] = False
    streamlines = streamlines[long_streamlines]

    return streamlines


def weighting_streamlines(
    out_folder_name,
    streamlines,
    bvec_file,
    weight_by="1.5_2_AxPasi5",
    hue=[0.0, 1.0],
    saturation=[0.0, 1.0],
    scale=[2, 7],
    fig_type="",
):
    """
    weight_by = '1.5_2_AxPasi5'
    hue = [0.0,1.0]
    saturation = [0.0,1.0]
    scale = [3,6]
    """
    from dipy.viz import window, actor
    from dipy.tracking.streamline import values_from_volume

    weight_by_data, affine = load_weight_by_img(bvec_file, weight_by)

    stream = list(streamlines)
    vol_per_tract = values_from_volume(weight_by_data, stream, affine=affine)

    pfr_data = load_weight_by_img(bvec_file, "1.5_2_AxFr5")[0]

    pfr_per_tract = values_from_volume(pfr_data, stream, affine=affine)

    # Leave out from the calculation of mean value per tract, a chosen quantile:
    vol_vec = weight_by_data.flatten()
    q = np.quantile(vol_vec[vol_vec > 0], 0.95)
    mean_vol_per_tract = []
    for s, pfr in zip(vol_per_tract, pfr_per_tract):
        s = np.asanyarray(s)
        non_out = [s < q]
        pfr = np.asanyarray(pfr)
        high_pfr = [pfr > 0.5]
        mean_vol_per_tract.append(np.nanmean(s[tuple(non_out and high_pfr)]))

    lut_cmap = actor.colormap_lookup_table(
        hue_range=hue, saturation_range=saturation, scale_range=scale
    )
    streamlines_actor = actor.line(
        streamlines, mean_vol_per_tract, linewidth=1, lookup_colormap=lut_cmap
    )
    bar = actor.scalar_bar(lut_cmap)
    r = window.Renderer()
    r.add(streamlines_actor)
    r.add(bar)
    # mean_pasi_weighted_img = out_folder_name+'\streamlines\mean_pasi_weighted' + fig_type + '.png'
    mean_pasi_weighted_img = f"{out_folder_name}/mean_pasi_weighted{fig_type}.png"
    # window.show(r)
    # r.set_camera(r.camera_info())
    r.set_camera(
        position=(-389.00, 225.24, 62.02),
        focal_point=(1.78, -3.27, -12.65),
        view_up=(0.00, -0.31, 0.95),
    )
    # window.record(r, out_path=mean_pasi_weighted_img, size=(800, 800))
    window.snapshot(r, fname=mean_pasi_weighted_img, size=(800, 800))
    return r


def load_ft(tract_path, nii_file):
    from dipy.io.streamline import load_trk, Space

    streams = load_trk(tract_path, nii_file, Space.RASMM)
    streamlines = streams.get_streamlines_copy()

    return streamlines


def save_ft(folder_name, subj_name, streamlines, nii_file, file_name="_wholebrain.trk"):
    from dipy.io.streamline import save_trk
    from dipy.io.stateful_tractogram import StatefulTractogram, Space

    # dir_name = folder_name + '\streamlines'
    # if not os.path.exists(dir_name):
    #     os.mkdir(dir_name)

    tract_name = f"{folder_name}/{subj_name}{file_name}"
    save_trk(StatefulTractogram(streamlines, nii_file, Space.RASMM), tract_name)


def nodes_by_index(folder_name, atlas_name="megaatlas"):
    import numpy as np
    import nibabel as nib

    # lab = folder_name + r'\rMegaAtlas_Labels_highres.nii'
    lab = os.path.join(folder_name, f"{atlas_name}_in_subj_dwi.nii.gz")
    lab_file = nib.load(lab)
    lab_labels = lab_file.get_data()
    affine = lab_file.affine
    uni = np.unique(lab_labels)
    lab_labels_index = lab_labels
    for index, i in enumerate(uni):
        lab_labels_index[lab_labels_index == i] = index
    return lab_labels_index, affine


def nodes_by_index_mega(folder_name):
    import nibabel as nib

    #    lab = folder_name + r'\rMegaAtlas_Labels.nii'
    lab = os.path.join(folder_name, r"LabelsAtlas2Highres_resampled.nii.gz")
    lab_file = nib.load(lab)
    lab_labels = lab_file.get_fdata()
    affine = lab_file.affine
    lab_labels_index = [labels for labels in lab_labels]
    lab_labels_index = np.asarray(lab_labels_index, dtype="int")
    return lab_labels_index, affine


def nodes_labels_mega(index_to_text_file):
    labels_file = open(index_to_text_file, "r", errors="ignore")
    labels_name = labels_file.readlines()
    labels_file.close()
    labels_table = []
    labels_headers = []
    idx = []
    for line in labels_name:
        if not line[0] == "#":
            labels_table.append([col for col in line.split("\t") if col])
        elif "ColHeaders" in line:
            labels_headers = [col for col in line.split(" ") if col]
            labels_headers = labels_headers[2:]
    for l in labels_table:
        head = l[1]
        labels_headers.append(head[:-1])
        idx.append(int(l[0]) - 1)
    return labels_headers, idx


def non_weighted_con_mat_mega(
    streamlines, lab_labels_index, affine, idx, folder_name, fig_type=""
):
    from dipy.tracking import utils
    import numpy as np

    if len(fig_type) >> 0:
        fig_type = "_" + fig_type

    m, grouping = utils.connectivity_matrix(
        streamlines,
        affine,
        lab_labels_index,
        return_mapping=True,
        mapping_as_streamlines=True,
    )
    mm = m[1:]
    mm = mm[:, 1:]
    mm = mm[idx]
    mm = mm[:, idx]
    new_data = (
        1 / mm
    )  # values distribute between 0 and 1, 1 represents distant nodes (only 1 tract)
    # new_data[new_data > 1] = 2
    np.save(f"{folder_name}/non-weighted_mega{fig_type}", new_data)
    np.save(f"{folder_name}/non-weighted_mega{fig_type}_nonnorm", mm)

    return new_data, m, grouping


def weighted_con_mat_mega(
    bvec_file, weight_by, grouping, idx, folder_name, fig_type=""
):
    from dipy.tracking.streamline import values_from_volume
    import numpy as np

    if len(fig_type) >> 0:
        fig_type = "_" + fig_type

    weight_by_data, affine = load_weight_by_img(bvec_file, weight_by)

    pfr_data = load_weight_by_img(bvec_file, "1.5_2_AxFr5")[0]

    vol_vec = weight_by_data.flatten()
    q = np.quantile(vol_vec[vol_vec > 0], 0.95)
    m_weighted = np.zeros((len(idx), len(idx)), dtype="float64")
    for pair, tracts in grouping.items():
        if pair[0] == 0 or pair[0] > len(idx) or pair[1] == 0 or pair[1] > len(idx):
            continue
        else:
            mean_vol_per_tract = []
            vol_per_tract = values_from_volume(weight_by_data, tracts, affine=affine)
            pfr_per_tract = values_from_volume(pfr_data, tracts, affine=affine)
            for s, pfr in zip(vol_per_tract, pfr_per_tract):
                s = np.asanyarray(s)
                non_out = [s < q]
                pfr = np.asanyarray(pfr)
                high_pfr = [pfr > 0.5]
                mean_vol_per_tract.append(np.nanmean(s[tuple(non_out and high_pfr)]))
            mean_path_vol = np.nanmean(mean_vol_per_tract)
            m_weighted[pair[0] - 1, pair[1] - 1] = mean_path_vol
            m_weighted[pair[1] - 1, pair[0] - 1] = mean_path_vol

    mm_weighted = m_weighted[idx]
    mm_weighted = mm_weighted[:, idx]
    # mm_weighted[mm_weighted<0.01] = 0
    # new_data = (10-mm_weighted)/10 # normalization between 0 and 1
    new_data = 1 / (mm_weighted * 8.75)  # 8.75 - axon diameter 2 ACV constant
    # new_data[new_data ==1] = 2
    # np.save(folder_name + r'\weighted_mega'+fig_type, new_data)
    # np.save(folder_name + r'\weighted_mega'+fig_type+'_nonnorm', mm_weighted)
    np.save(f"{folder_name}/weighted_mega{fig_type}", new_data)
    np.save(f"{folder_name}/weighted_mega{fig_type}_nonnorm", mm_weighted)

    return new_data, mm_weighted


def draw_con_mat(data, h, fig_name, is_weighted=False):
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np

    max_val = np.max(data[np.isfinite(data)])
    data[~np.isfinite(data)] = np.nan
    if is_weighted:
        # data[data>0.8*max_val] = 0.8*max_val
        data[~np.isfinite(data)] = max_val
        mat_title = "AxCaliber weighted connectivity matrix"
        plt.figure(1, [30, 24])
        cmap = cm.YlOrRd
        cmap.set_over("black")
        plt.imshow(
            data,
            interpolation="nearest",
            cmap=cmap,
            origin="upper",
            vmax=0.99 * max_val,
        )
        plt.colorbar()
        plt.xticks(ticks=np.arange(0, len(data), 1), labels=h)
        plt.yticks(ticks=np.arange(0, len(data), 1), labels=h)
        plt.title(mat_title, fontsize=32)
        plt.tick_params(axis="x", pad=10.0, labelrotation=90, labelsize=11)
        plt.tick_params(axis="y", pad=10.0, labelsize=11)
        plt.savefig(fig_name)
        plt.show()
    else:
        data[~np.isfinite(data)] = max_val
        mat_title = "Number of tracts weighted connectivity matrix"
        plt.figure(1, [30, 24])
        cmap = cm.YlOrRd
        cmap.set_over("black")
        plt.imshow(
            data,
            interpolation="nearest",
            cmap=cmap,
            origin="upper",
            vmax=0.99 * max_val,
            norm=colors.LogNorm(vmax=0.99 * max_val),
        )
        plt.colorbar()
        plt.xticks(ticks=np.arange(0, len(data), 1), labels=h)
        plt.yticks(ticks=np.arange(0, len(data), 1), labels=h)
        plt.title(mat_title, fontsize=32)
        plt.tick_params(axis="x", pad=10.0, labelrotation=90, labelsize=11)
        plt.tick_params(axis="y", pad=10.0, labelsize=11)
        plt.savefig(fig_name)
        plt.show()


def weighted_connectivity_matrix_mega(
    streamlines,
    folder_name,
    bvec_file,
    fig_type="whole brain",
    weight_by="1.5_2_AxPasi5",
):

    lab_labels_index, affine = nodes_by_index_mega(folder_name)

    index_to_text_file = r"C:\Users\Admin\my_scripts\aal\megaatlas\megaatlas2nii.txt"
    labels_headers, idx = nodes_labels_mega(index_to_text_file)

    # non-weighted:

    new_data, m, grouping = non_weighted_con_mat_mega(
        streamlines, lab_labels_index, affine, idx, folder_name, fig_type
    )
    h = labels_headers
    fig_name = folder_name + r"\non-weighted(" + fig_type + ", MegaAtlas).png"
    draw_con_mat(new_data, h, fig_name, is_weighted=False)

    # weighted:

    new_data, mm_weighted = weighted_con_mat_mega(
        bvec_file, weight_by, grouping, idx, folder_name, fig_type
    )
    fig_name = folder_name + r"\Weighted(" + fig_type + ", MegaAtlas).png"
    draw_con_mat(new_data, h, fig_name, is_weighted=True)


def load_weight_by_img(bvec_file, weight_by):
    import nibabel as nib

    weight_by_file = bvec_file[:-5:] + "_" + weight_by + ".nii"
    weight_by_img = nib.load(weight_by_file)
    weight_by_data = weight_by_img.get_data()
    affine = weight_by_img.affine

    return weight_by_data, affine


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    subj = all_subj_folders
    names = all_subj_names
    # masks = ['cc_cortex_cleaned','genu_cortex_cleaned','body_cortex_cleaned','splenium_cortex_cleaned']
    # masks = ['cc','genu','body','splenium']

    for s, n in zip(subj, names):
        folder_name = r"C:\Users\Admin\my_scripts\Ax3D_Pack\V6\after_file_prep" + s
        dir_name = folder_name + "\streamlines"
        gtab, data, affine, labels, white_matter, nii_file, bvec_file = load_dwi_files(
            folder_name
        )
        lab_labels_index = nodes_by_index_mega(folder_name)[0]
        # csd_fit = create_csd_model(data, gtab, white_matter, sh_order=6)
        # fa, classifier = create_fa_classifier(gtab, data, white_matter)
        # seeds = create_seeds(folder_name, lab_labels_index, affine, use_mask=False, mask_type='cc', den=4)
        # streamlines = create_streamlines(csd_fit, classifier, seeds, affine)
        # save_ft(folder_name,n,streamlines,nii_file, file_name="_fx_5d_plus_new.trk")
        tract_path = folder_name + r"\streamlines" + n + "_wholebrain_1d_plus.trk"
        streamlines = load_ft(tract_path, nii_file)
        weighted_connectivity_matrix_mega(
            streamlines,
            folder_name,
            bvec_file,
            fig_type="wholebrain_plus_new2",
            weight_by="1.5_2_AxPasi5",
        )

        # for m in masks:
        #    seeds = create_seeds(folder_name, lab_labels_index, affine, use_mask=True, mask_type=m, den=7)
        #    streamlines = create_streamlines(csd_fit, classifier, seeds, affine)
        #    save_ft(folder_name, n, streamlines, nii_file, file_name="_" + m + "_7d.trk")
