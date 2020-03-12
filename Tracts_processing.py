import os
from dipy.io.streamline import load_tractogram
import glob
from Networks_plasticity import (
    Mrtrix3_methods as mrt_methods,
    Preprocessing as Prep,
    weighted_tracts,
    Preproc_Methods as Methods,
)
import shutil

MOTHER_DIR = "/home/gal/Brain_Networks"
ATLAS_DIR = "/home/gal/Atlases"


def init_process(mother_dir="/home/gal/Brain_Networks"):
    prep = Prep.Preprocess(mother_dir)
    subjects = prep.subjects
    return mother_dir, subjects


class Generate_Tracts_with_dipy:
    def __init__(self, subject=None):
        self.mother_dir, self.subjects = init_process()
        if subject:
            self.subjects = subject

    def Gen_init_tracts_dipy(self, subj):
        folder_name = f"{self.mother_dir}/Niftis/{subj}"
        (
            gtab,
            data,
            affine,
            labels,
            white_matter,
            nii_file,
            bvec_file,
        ) = weighted_tracts.load_dwi_files(folder_name=folder_name)
        seeds = weighted_tracts.create_seeds_new(labels=labels, affine=affine)
        csd_fit = weighted_tracts.create_csd_model(
            data=data, gtab=gtab, white_matter=white_matter
        )
        fa, classifier = weighted_tracts.create_fa_classifier(
            gtab=gtab, data=data, white_matter=white_matter
        )
        streamlines = weighted_tracts.create_streamlines(
            csd_fit=csd_fit, classifier=classifier, seeds=seeds, affine=affine
        )
        return streamlines, nii_file

    def save_tracts_file(self, subj, streamlines, nii_file):
        folder_name = f"{self.mother_dir}/Derivatives/Streamlines/{subj}"
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)
        weighted_tracts.save_ft(folder_name, subj, streamlines, nii_file)

    def run_whole_head_tractography(self):
        for subj in self.subjects:
            streamlines, nii_file = self.Gen_init_tracts_dipy(subj)
            self.save_tracts_file(subj, streamlines, nii_file)


class Generate_Connectivity_dipy:
    def __init__(self, subject=None):
        self.mother_dir, self.subjects = init_process()
        if subject:
            self.subjects = [subject]

    def init_subject_params(self, subj: str):
        dwi_file = f"{self.mother_dir}/Niftis/{subj}/dwi/diff_corrected.nii.gz"
        bvec_file = glob.glob(f"{self.mother_dir}/Niftis/{subj}/dwi/*.bvec")[0]
        reg_folder = (
            f"{self.mother_dir}/Derivatives/Registrations/{subj}/Atlases_and_Transforms"
        )
        stream_folder = f"{self.mother_dir}/Derivatives/Streamlines/{subj}"
        streamlines_file = f"{stream_folder}/{subj}_wholebrain.trk"
        streamlines = load_tractogram(streamlines_file, dwi_file)

        return streamlines, stream_folder, reg_folder, bvec_file

    def run_whole_head_connectivity(self):
        atlas_folder = f"{self.mother_dir}/Derivatives/megaatlas"
        for subj in self.subjects:
            (
                streamlines,
                stream_folder,
                reg_folder,
                bvec_file,
            ) = self.init_subject_params(subj)
            lab_labels_index, affine = weighted_tracts.nodes_by_index(reg_folder)
            index_file = f"{atlas_folder}/megaatlascortex2nii_origin.txt"
            labels_headers, idx = weighted_tracts.nodes_labels_mega(index_file)
            new_data, m, grouping = weighted_tracts.non_weighted_con_mat_mega(
                streamlines.streamlines, lab_labels_index, affine, idx, stream_folder
            )
            non_weighted_fig_name = (
                f"{stream_folder}/Whole_Head_non-weighted_Connectivity.jpg"
            )
            weighted_tracts.draw_con_mat(
                new_data, labels_headers, non_weighted_fig_name
            )
            weight_by = "1.5_2_AxPasi5"
            weighted_fig_name = (
                f"{stream_folder}/Whole_Head_-{weight_by}_weighted_Connectivity.jpg"
            )
            new_data, mm_weighted = weighted_tracts.weighted_con_mat_mega(
                bvec_file, weight_by, grouping, idx, stream_folder
            )
            weighted_tracts.draw_con_mat(
                new_data, labels_headers, weighted_fig_name, is_weighted=True
            )


class Generate_Tracts_with_mrtrix3:
    def __init__(self, mother_dir=MOTHER_DIR, subj=None):
        self.mother_dir = mother_dir
        if subj:
            subjects_dirs = [f"{mother_dir}/Derivatives/dwi_prep/{subj}"]
            subjects = [subj]
        else:
            subjects_dirs = glob.glob(f"{mother_dir}/Derivatives/dwi_prep/sub*")
            subjects = [subj.split(os.sep)[-1] for subj in subjects_dirs]
        subjects_dirs.sort()
        subjects.sort()
        self.subjects = subjects
        self.subjects_dirs = subjects_dirs

    def load_files(self, folder_name: str):
        subj = folder_name.split(os.sep)[-1]
        dwi_file = f"{folder_name}/{subj}_dwi.mif"
        dwi_mask = glob.glob(f"{folder_name}/*dwi_brain_mask.mif")[0]
        anat_dir = f"{folder_name}/T1.anat"
        corr_list = glob.glob(f"{anat_dir}/T1_biascorr_*mif")
        corr_list.sort()
        bias_corr_brain, bias_corr_mask = corr_list
        return dwi_file, dwi_mask, bias_corr_brain, bias_corr_mask

    def Gen_FA(self, dwi_file: str, mask_file: str):
        dti_file = f"{os.path.dirname(dwi_file)}/dti.mif"
        fa_file = f"{os.path.dirname(dwi_file)}/fa.mif"
        if not os.path.isfile(dti_file) or not os.path.isfile(fa_file):
            print("Generating FA image for group-level analysis")
            fa_file = mrt_methods.fit_tensors(dwi_file, mask_file, dti_file)
        return fa_file

    def gen_Response(self, dwi_file: str, dwi_mask: str):
        working_dir = os.path.dirname(dwi_file)
        wm_resp, gm_resp, csf_resp = [
            f"{working_dir}/response_{tissue}.txt" for tissue in ["wm", "gm", "csf"]
        ]
        if not os.path.isfile(wm_resp):
            print("Estimating tissue response functions for spherical deconvolution")
            wm_resp, gm_resp, csf_resp = mrt_methods.gen_response(
                dwi_file, dwi_mask, working_dir
            )
        return wm_resp, gm_resp, csf_resp

    def fibre_orientation(
        self, dwi_file: str, dwi_mask: str, wm_resp: str, gm_resp: str, csf_resp: str
    ):
        fod_wm, fod_gm, fod_csf = [
            resp.replace("response", "FOD") for resp in [wm_resp, gm_resp, csf_resp]
        ]
        fod_wm, fod_gm, fod_csf = [
            odf.replace("txt", "mif") for odf in [fod_wm, fod_gm, fod_csf]
        ]
        tissues = f"{os.path.dirname(dwi_file)}/tissues.mif"
        if not os.path.isfile(tissues):
            print("Estimating Fibre Orientation Distributions")
            fod_wm, fod_gm, fod_csf = mrt_methods.calc_fibre_orientation(
                dwi_file, dwi_mask, wm_resp, gm_resp, csf_resp
            )
            tissues = mrt_methods.gen_tissues_orient(fod_wm, fod_gm, fod_csf)
        return fod_wm, fod_gm, fod_csf, tissues

    def gen_dwi_to_T1_contrast(
        self, dwi_file: str, dwi_mask: str, t1_file: str, t1_mask: str
    ):
        meanbzero = dwi_file.replace(".mif", "_meanbzero.mif")
        dwi_pseudoT1 = dwi_file.replace(".mif", "_pseudoT1.mif")
        T1_pseudobzero = f"{os.path.dirname(dwi_file)}/T1_pseudobzero.mif"
        if not os.path.isfile(T1_pseudobzero):
            print(
                "Generating contrast-matched images for inter-modal registration between DWIs and T1"
            )
            meanbzero, dwi_pseudoT1, T1_pseudobzero = mrt_methods.DWI_to_T1_cont(
                dwi_file, dwi_mask, t1_file, t1_mask
            )
        return meanbzero, dwi_pseudoT1, T1_pseudobzero

    def reg_dwi_and_t1(
        self,
        dwi_file: str,
        t1_brain: str,
        dwi_pseudoT1: str,
        T1_pseudobzero: str,
        meanbzero: str,
        t1_mask: str,
        dwi_mask: str,
    ):
        working_dir = os.path.dirname(dwi_file)
        t1_registered = f"{working_dir}/T1_registered.mif"
        t1_mask_registered = f"{working_dir}/T1_mask_registered.mif"
        if not os.path.isfile(t1_registered) or not os.path.isfile(t1_mask_registered):
            print("Performing registration between DWIs and T1")
            t1_registered, t1_mask_registered = mrt_methods.reg_dwi_T1(
                dwi_file,
                t1_brain,
                dwi_pseudoT1,
                T1_pseudobzero,
                meanbzero,
                t1_mask,
                dwi_mask,
                True,
            )
        else:
            t1_registered, t1_mask_registered = mrt_methods.reg_dwi_T1(
                dwi_file,
                t1_brain,
                dwi_pseudoT1,
                T1_pseudobzero,
                meanbzero,
                t1_mask,
                dwi_mask,
                False,
            )
        return t1_registered, t1_mask_registered

    def gen_5tt(self, t1_registered: str, t1_mask_regisitered: str):
        out_vis = f"{os.path.dirname(t1_registered)}/vis.mif"
        out_5tt = f"{os.path.dirname(t1_registered)}/5TT.mif"
        if not os.path.isfile(out_vis):
            print(
                "Generating five-tissue-type (5TT) image for Anatomically-Constrained Tractography (ACT)"
            )
            out_vis, out_5tt = mrt_methods.five_tissue(
                t1_registered, t1_mask_regisitered
            )
        return out_vis, out_5tt

    def calc_tracts(self, fod_wm: str, seg_5tt: str):
        Tracts = f"{os.path.dirname(fod_wm)}/tractogram.tck"
        if not os.path.isfile(Tracts):
            print("Performing whole-brain fibre-tracking")
            Tracts = mrt_methods.generate_tracks(fod_wm, seg_5tt)
        return Tracts

    def convert_tck_to_trk(self, tck_file: str, dwi_file: str):
        trk_file = tck_file.replace("tck", "trk")
        if not os.path.isfile(trk_file):
            print("Converting tractography file from .tck format to .trk")
            trk_file = mrt_methods.convert_tck_to_trk(tck_file, dwi_file)
        return trk_file

    def sort_files(self, folder_name):
        connectome, dwi, tractogram, anat = [
            f"{folder_name}/{sub_dir}"
            for sub_dir in ["connectome", "dwi", "tractogram", "anat"]
        ]
        for sub_dir in [connectome, dwi, tractogram, anat]:
            if not os.path.isdir(sub_dir):
                os.makedirs(sub_dir)
        new_file = None
        for file in glob.glob(f"{folder_name}/*"):
            if not os.path.isdir(file):
                header = file.split(os.sep)[-1]
                if (
                    "dwi" in header
                    or "AP" in header
                    or "b0" in header
                    or "tissues" in header
                    or "bzero" in header
                ):
                    new_file = f'{dwi}/{header.replace("mif","nii")}'
                    if "mif" in file and not os.path.isfile(new_file):
                        new_file = mrt_methods.convert_to_mif(file, new_file)
                    else:
                        shutil.copy(file, f"{dwi}/{header}")
                elif "T1" in file or "vis" in file or "5TT" in file:
                    new_file = f'{anat}/{file.split(os.sep)[-1].replace("mif", "nii")}'
                    if "mif" in file and not os.path.isfile(new_file):
                        new_file = mrt_methods.convert_to_mif(file, new_file)
                    else:
                        shutil.copy(file, f"{anat}/{file.split(os.sep)[-1]}")
                else:
                    new_file = (
                        f'{tractogram}/{file.split(os.sep)[-1].replace("mif", "nii")}'
                    )
                    if "mif" in file and not os.path.isfile(new_file):
                        new_file = mrt_methods.convert_to_mif(file, new_file)
                    else:
                        shutil.copy(file, f"{tractogram}/{file.split(os.sep)[-1]}")
                if new_file:
                    if os.path.isfile(new_file):
                        print(
                            f"Moved {file.split(os.sep)[-1]} to {os.sep.join(new_file.split(os.sep)[-2:])}"
                        )
                        # os.remove(file)

    def run(self):
        for folder_name in self.subjects_dirs:
            dwi_file, dwi_mask, T1_brain, T1_mask = self.load_files(folder_name)
            fa_file = self.Gen_FA(dwi_file, dwi_mask)
            wm_resp, gm_resp, csf_resp = self.gen_Response(dwi_file, dwi_mask)
            fod_wm, fod_gm, fod_csf, tissues = self.fibre_orientation(
                dwi_file, dwi_mask, wm_resp, gm_resp, csf_resp
            )
            meanbzero, dwi_pseudoT1, T1_pseudobzero = self.gen_dwi_to_T1_contrast(
                dwi_file, dwi_mask, T1_brain, T1_mask
            )
            t1_registered, t1_mask_registered = self.reg_dwi_and_t1(
                dwi_file,
                T1_brain,
                dwi_pseudoT1,
                T1_pseudobzero,
                meanbzero,
                T1_mask,
                dwi_mask,
            )
            vis_5tt, seg_5tt = self.gen_5tt(t1_registered, t1_mask_registered)
            Tracts_tck = self.calc_tracts(fod_wm, seg_5tt)
            Tracts_trk = self.convert_tck_to_trk(Tracts_tck, t1_registered)
            self.sort_files(folder_name)


class Generate_subjects_atlas:
    def __init__(self, mother_dir=MOTHER_DIR, subj=None, atlas_name: str = "megaatlas"):
        self.mother_dir = mother_dir
        if subj:
            subjects_dirs = [f"{mother_dir}/Derivatives/dwi_prep/{subj}"]
            subjects = [subj]
        else:
            subjects_dirs = glob.glob(f"{mother_dir}/Derivatives/dwi_prep/sub*")
            subjects = [subj.split(os.sep)[-1] for subj in subjects_dirs]
        subjects_dirs.sort()
        subjects.sort()
        self.subjects = subjects
        self.subjects_dirs = subjects_dirs
        self.atlas_name = atlas_name
        self.atlas_dir = f"{ATLAS_DIR}/{atlas_name}"

    def init_atlas_files(self):
        for file in os.listdir(self.atlas_dir):
            if "highres.nii" in file:
                highres_file = f"{self.atlas_dir}/{file}"
            elif ".txt" in file:
                labels = f"{self.atlas_dir}/{file}"
        return highres_file, labels

    def init_subj_dirs(self, subj: str):
        deriv = f"{self.mother_dir}/Derivatives"
        transforms_dir = f"{deriv}/Atlases_and_transforms/{subj}"
        if not os.path.isdir(transforms_dir):
            os.makedirs(transforms_dir)
        return transforms_dir

    def init_subject_files(self, folder_name: str):
        subj = folder_name.split(os.sep)[-1]
        T1_biascorr_brain = f"{folder_name}/T1.anat/T1_biascorr_brain.nii.gz"
        T1_registered = f"{folder_name}/anat/T1_registered.nii"
        mni2T1_warp = f"{folder_name}/T1.anat/MNI_to_T1_nonlin_field.nii.gz"
        dwi_meanbzero = f"{folder_name}/dwi/{subj}_dwi_meanbzero.nii"
        return T1_biascorr_brain, T1_registered, mni2T1_warp, dwi_meanbzero

    def atlas2subj_nonlin(
        self, highres_atlas: str, ref: str, field_file: str, transforms_dir: str
    ):
        out_file = f"{transforms_dir}/{self.atlas_name}_in_subj_T1.nii.gz"
        if not os.path.isfile(out_file):
            print(
                "Applying non-linear transformation on atlas, to convert it to subject`s T1 space"
            )
            aw = Methods.apply_warp(highres_atlas, out_file, field_file, ref)
            aw.run()
        return out_file

    def atlas2subj_lin(self, atlas_in_T1: str, dwi_meanbzero: str, transforms_dir: str):
        out_file = f"{transforms_dir}/{self.atlas_name}_in_subj_dwi.nii.gz"
        if not os.path.isfile(out_file):
            print(
                "Performing linear transformation on subject`s T1 atlas, to convert it to subject`s DWI space"
            )
            aw = Methods.apply_XFM(atlas_in_T1, dwi_meanbzero, out_file)
            aw.run()
        return out_file

    def run(self):
        highres_atlas, labels = self.init_atlas_files()
        for subnum, folder_name in enumerate(self.subjects_dirs):
            subj = self.subjects[subnum]
            transforms_dir = self.init_subj_dirs(subj)
            (
                T1_biascorr_brain,
                T1_registered,
                mni2T1_warp,
                dwi_meanbzero,
            ) = self.init_subject_files(folder_name)
        atlas_in_T1 = self.atlas2subj_nonlin(
            highres_atlas, T1_biascorr_brain, mni2T1_warp, transforms_dir
        )
        atlas_in_dwi = self.atlas2subj_lin(atlas_in_T1, dwi_meanbzero, transforms_dir)

        #################### FINISH ATLAS REGISTRATION ################


class Generate_Connectivity_mrtrix:
    def __init__(self, mother_dir=MOTHER_DIR, subj=None, atlas_name="megaatlas"):
        self.mother_dir = mother_dir
        self.atlas_name = atlas_name
        if subj:
            subjects_dirs = [f"{mother_dir}/Derivatives/dwi_prep/{subj}"]
            subjects = [subj]
        else:
            subjects_dirs = glob.glob(f"{mother_dir}/Derivatives/dwi_prep/sub*")
            subjects = [subj.split(os.sep)[-1] for subj in subjects_dirs]
        subjects_dirs.sort()
        subjects.sort()
        self.subjects = subjects
        self.subjects_dirs = subjects_dirs

    def init_subject_params(self, folder_name):
        subj = folder_name.split(os.sep)[-1]
        dwi_file = f"{folder_name}/dwi/{subj}_dwi.nii"
        t1_registered = f"{folder_name}/anat/T1_registered.nii"
        bvec_file = glob.glob(f"{self.mother_dir}/Niftis/{subj}/dwi/*.bvec")[0]
        reg_folder = f"{self.mother_dir}/Derivatives/Atlases_and_transforms/{subj}"
        stream_folder = f"{folder_name}/tractogram"
        connectome_folder = f"{folder_name}/connectome"
        streamlines_file = f"{stream_folder}/tractogram.trk"
        streamlines = load_tractogram(streamlines_file, t1_registered)

        return streamlines, stream_folder, reg_folder, bvec_file, connectome_folder

    def run_whole_head_connectivity(self):
        atlas_folder = f"/home/gal/Atlases/megaatlas"
        for i, folder_name in enumerate(self.subjects_dirs):
            subj = self.subjects[i]
            (
                streamlines,
                stream_folder,
                reg_folder,
                bvec_file,
                connectme_folder,
            ) = self.init_subject_params(folder_name)
            lab_labels_index, affine = weighted_tracts.nodes_by_index(reg_folder)
            index_file = f"{atlas_folder}/megaatlascortex2nii_origin.txt"
            labels_headers, idx = weighted_tracts.nodes_labels_mega(index_file)
            new_data, m, grouping = weighted_tracts.non_weighted_con_mat_mega(
                streamlines.streamlines, lab_labels_index, affine, idx, connectme_folder
            )
            non_weighted_fig_name = (
                f"{connectme_folder}/Whole_Head_non-weighted_Connectivity.jpg"
            )
            weighted_tracts.draw_con_mat(
                new_data, labels_headers, non_weighted_fig_name
            )
            weight_by = "1.5_2_AxPasi5"
            weighted_fig_name = (
                f"{connectme_folder}/Whole_Head_-{weight_by}_weighted_Connectivity.jpg"
            )
            new_data, mm_weighted = weighted_tracts.weighted_con_mat_mega(
                bvec_file, weight_by, grouping, idx, connectme_folder
            )
            weighted_tracts.draw_con_mat(
                new_data, labels_headers, weighted_fig_name, is_weighted=True
            )
