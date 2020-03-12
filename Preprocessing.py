import glob, os, shutil
from Networks_plasticity import (
    Mrtrix3_methods as MRT_Methods,
    Preproc_Methods as Methods,
)

MOTHER_DIR = "/home/gal/Brain_Networks"


class Preprocess:
    def __init__(self, mother_dir: str = MOTHER_DIR, data_type: str = "dwi"):
        self.mother_dir = mother_dir
        subjects_dirs = glob.glob(f"{mother_dir}/Niftis/sub*")
        subjects = [subj.split(os.sep)[-1] for subj in subjects_dirs]
        subjects.sort()
        self.subjects = subjects

    def Highres_brain_extract(self, subj):
        print(f"Performing brain segmentation for {subj}.\n")
        in_file = f"{self.mother_dir}/Niftis/{subj}/anat/{subj}_T1w.nii.gz"
        seg = Methods.cat12(in_file=in_file)
        seg.run()

    def create_tmp_dir(self, subj):
        if not os.path.isdir(f"{self.mother_dir}/Niftis/{subj}/dwi/tmp"):
            os.mkdir(f"{self.mother_dir}/Niftis/{subj}/dwi/tmp")

    def MotionCorrection(self, subj):
        print(f"Performing motion correction for {subj}.\n")
        in_file = f"{self.mother_dir}/Niftis/{subj}/dwi/{subj}_AP_dwi.nii.gz"
        mot_cor = Methods.MotionCorrection(in_file=in_file)
        mot_corrected = mot_cor.run()

    def Prepare4Eddy(self, subj):
        working_dir = f"{self.mother_dir}/Niftis/{subj}"
        prep_eddy = Methods.PrepareEddy(working_dir)
        topup_res, brain_mask, idx_file, acq = prep_eddy.run()
        field_coef = topup_res + "_fieldcoef.nii.gz"
        bvec = glob.glob(f"{working_dir}/dwi/*.bvec")[0]
        bval = glob.glob(f"{working_dir}/dwi/*.bval")[0]
        return field_coef, brain_mask, idx_file, bvec, bval, acq

    def RunEddy(self, fieldcoef, brain_mask, idx_file, bvec, bval, acq):
        in_file = bvec.replace(".bvec", "_motion_corrected.nii.gz")
        eddy = Methods.EddyCorrect(
            in_file=in_file,
            fieldcoef=fieldcoef,
            brain_mask=brain_mask,
            idx_file=idx_file,
            bvec=bvec,
            bval=bval,
            acq=acq,
        )
        eddy.run()

    def run_subjects_preproc(self):
        subjects = self.subjects
        for subj in subjects:
            if not os.path.isfile(
                f"{self.mother_dir}/Niftis/{subj}/anat/mri/p0{subj}_T1w.nii"
            ):
                self.Highres_brain_extract(subj)
            else:
                print(f"{subj} already went through cat12 brain segmentation.")
            self.create_tmp_dir(subj)
            if not os.path.isfile(
                f"{self.mother_dir}/Niftis/{subj}/dwi/{subj}_AP_dwi_motion_corrected.nii.gz"
            ):
                self.MotionCorrection(subj)
            else:
                print(f"{subj} already went through motion correction")
            # if not os.path.isfile(f'{self.mother_dir}/Niftis/{subj}/dwi/tmp/topup_res_fieldcoef.nii.gz'):
            field_coef, brain_mask, idx_file, bvec, bval, acq = self.Prepare4Eddy(subj)

            # if not os.path.isfile(f'{self.mother_dir}//Niftis/{subj}/dwi/diff_corrected.nii.gz'):
            self.RunEddy(field_coef, brain_mask, idx_file, bvec, bval, acq)
            eddy_res = f"{self.mother_dir}/Niftis/{subj}/dwi/tmp/diff_corrected.nii.gz"
            os.rename(
                eddy_res, f"{self.mother_dir}/Niftis/{subj}/dwi/diff_corrected.nii.gz"
            )

    def run_subjects_registrations(self):
        subjects = self.subjects
        atlas_dir = f"{self.mother_dir}/Derivatives/megaatlas"
        for subj in subjects:
            print(f"Creating atlas image for {subj}.")
            sub_dir = f"{self.mother_dir}/Niftis/{subj}"
            MPRAGE = f"{sub_dir}/anat/{subj}_T1w.nii.gz"
            bet = Methods.run_BET(in_file=MPRAGE)
            bet.run()
            subject_atlas = Methods.Create_Atlas(
                atlas_dir=atlas_dir, target_dir=sub_dir
            )
            subject_atlas.run()
            gen_wm = Methods.GenerateWhiteMatterMask(sub_dir=sub_dir)
            gen_wm.run()


class Mrtrix_prep:
    def __init__(self, mother_dir: str = MOTHER_DIR, subj=None, data_type: str = "dwi"):
        self.mother_dir = mother_dir
        self.data_type = data_type
        if subj:
            subjects_dirs = [f"{mother_dir}/Niftis/{subj}"]
            subjects = [subj]
        else:
            subjects_dirs = glob.glob(f"{mother_dir}/Niftis/sub*")
            subjects = [subj.split(os.sep)[-1] for subj in subjects_dirs]
        subjects_dirs.sort()
        subjects.sort()
        self.subjects = subjects
        self.subjects_dirs = subjects_dirs

    def MotionCorrection_and_Mask(self, in_file, folder: str):
        mot_corrected = f"{folder}/{in_file.split(os.sep)[-1]}"
        if not os.path.isfile(mot_corrected):
            print("Performing Motion Correction and Brain Extraction.")
            mot_cor = Methods.MotionCorrection(in_file=in_file, out_file=mot_corrected)
            mot_corrected = mot_cor.run()
            mot_cor_mask = Methods.run_BET(mot_corrected)
            mot_cor_mask = mot_cor_mask.run()
        else:
            mot_cor_mask = mot_corrected.replace(".nii.gz", "_brain_mask.nii.gz")
        return mot_corrected, mot_cor_mask

    def init_mrt_folder(self, folder_name):
        mrt_folder = MRT_Methods.init_mrt_folder(folder_name, "dwi")
        anat_file, dwi_file, bvec, bval, PA_file = MRT_Methods.load_initial_files(
            folder_name, "dwi"
        )
        dwi_mask = self.MotionCorrection_and_Mask(dwi_file, mrt_folder)
        file_list = [anat_file, dwi_file, PA_file, dwi_mask]
        print("Converting files into .mif format...")
        for f in file_list:
            f_name = f.split(os.sep)[-1]
            new_f = f'{mrt_folder}/{f_name.replace(".nii.gz",".mif")}'
            if not os.path.isfile(new_f):
                if "T1" in f:
                    print("Importing T1 image into temporary directory")
                    new_anat = MRT_Methods.convert_to_mif(f, new_f)
                elif "mask" in f:
                    print("Importing mask image into temporary directory")
                    new_mask = MRT_Methods.convert_to_mif(f, new_f)
                else:
                    if "AP" in f:
                        print("Importing DWI data into temporary directory")
                        new_dwi = MRT_Methods.convert_to_mif(f, new_f, bvec, bval)
                    elif "PA" in f:
                        print(
                            "Importing reversed phased encode data into temporary directory"
                        )
                        new_PA = MRT_Methods.convert_to_mif(f, new_f)
            else:
                if "T1" in f:
                    new_anat = new_f
                elif "mask" in f:
                    new_mask = new_f
                else:
                    if "AP" in f:
                        new_dwi = new_f
                    elif "PA" in f:
                        new_PA = new_f
        return new_anat, new_dwi, new_mask, new_PA

    def init_func_folder(self, folder_name):
        mrt_folder = MRT_Methods.init_mrt_folder(folder_name, "func")
        anat_file, func, sbref, AP, PA, bvec, bval = MRT_Methods.load_initial_files(
            folder_name, "func"
        )
        dwi, mask = self.MotionCorrection_and_Mask(AP, mrt_folder)
        file_list = [anat_file, func, sbref, AP, PA, mask]
        print("Converting files into .mif format...")
        for f in file_list:
            f_name = f.split(os.sep)[-1]
            new_f = f'{mrt_folder}/{f_name.replace(".nii.gz",".mif")}'
            if not os.path.isfile(new_f):
                if "T1" in f:
                    print("Importing T1 image into temporary directory")
                    new_anat = MRT_Methods.convert_to_mif(f, new_f)
                elif "mask" in f:
                    print("Importing mask image into temporary directory")
                    new_mask = MRT_Methods.convert_to_mif(f, new_f)
                else:
                    if "AP" in f:
                        print("Importing DWI data into temporary directory")
                        new_dwi = MRT_Methods.convert_to_mif(f, new_f, bvec, bval)
                    elif "PA" in f:
                        print(
                            "Importing reversed phased encode data into temporary directory"
                        )
                        new_PA = MRT_Methods.convert_to_mif(f, new_f)
                    elif "rest_bold" in f:
                        print("Importing rs-fMRI data into temporary directory")
                        new_func = MRT_Methods.convert_to_mif(f, new_f)
                    elif "rest_sbref" in f:
                        print("Importing rs reference to temporary directory")
                        new_sbref = MRT_Methods.convert_to_mif(f, new_f)
            else:
                if "T1" in f:
                    new_anat = new_f
                elif "mask" in f:
                    new_mask = new_f
                else:
                    if "AP" in f:
                        new_dwi = new_f
                    elif "PA" in f:
                        new_PA = new_f
                    elif "rest_bold" in f:
                        new_func = new_f
                    elif "rest_sbref" in f:
                        new_sbref = new_f
        return new_anat, new_func, new_sbref, new_dwi, new_mask, new_PA

    def denoise(self, dwi_file: str, dwi_mask: str):
        out_file = dwi_file.replace(".mif", "_denoised.mif")
        if not os.path.isfile(out_file):
            print("Performing MP-PCA denoising of DWI data")
            out_file = MRT_Methods.Denoise(dwi_file, dwi_mask, out_file)
        return out_file

    def unring(self, denoised: str):
        out_file = denoised.replace(".mif", "_degibbs.mif")
        if not os.path.isfile(out_file):
            print("Performing Gibbs ringing removal for DWI data")
            MRT_Methods.Unring(in_file=denoised, out_file=out_file)
        return out_file

    def DWI_preproc(self, degibbs: str, PA: str, func: str = None):
        if not func:
            data_type = "dwi"
            func = None
            out_file = f"{os.path.dirname(degibbs)}/dwi1_preprocessed.mif"
        else:
            data_type = "func"
            func = func
            out_file = f"{os.path.dirname(degibbs)}/func_preprocessed.mif"
        if not os.path.isfile(out_file):
            print("Performing various geometric corrections of DWIs")
            out_file = MRT_Methods.DWI_prep(
                degibbs=degibbs,
                PA=PA,
                out_file=out_file,
                data_type=self.data_type,
                func=func,
            )
        return out_file

    def bias_correct(self, preprocessed: str, subj: str):
        out_file = f"{os.path.dirname(preprocessed)}/{subj}_dwi.mif"
        if not os.path.isfile(out_file):
            print("Performing initial B1 bias field correction of DWIs")
            out_file = MRT_Methods.bias_correct(preprocessed, out_file)
        return out_file

    def T1_correction(self, anat: str):
        anat_dir = f"{os.path.dirname(anat)}/T1.anat"
        if not os.path.isdir(anat_dir):
            print(
                "Performing brain extraction and B1 bias field correction of T1 image"
            )
            bias_corr_brain, bias_corr_mask = MRT_Methods.T1_correction(anat)
        else:
            corr_list = glob.glob(f"{anat_dir}/T1_biascorr_")
            corr_list.sort()
            bias_corr_brain, bias_corr_mask = corr_list
        return bias_corr_brain, bias_corr_mask

    def run(self):
        if "dwi" in self.data_type:
            for i, folder_name in enumerate(self.subjects_dirs):
                subj = self.subjects[i]
                print(f"Analyzing {subj}...")
                anat, dwi, mask, PA = self.init_mrt_folder(folder_name)
                denoised = self.denoise(dwi_file=dwi, dwi_mask=mask)
                degibbs = self.unring(denoised)
                preprocessed = self.DWI_preproc(degibbs, PA)
                dwi_file = self.bias_correct(preprocessed, subj=subj)
                bias_corr_brain, bias_corr_mask = MRT_Methods.T1_correction(anat)
        elif "func" in self.data_type:
            for i, folder_name in enumerate(self.subjects_dirs):
                subj = self.subjects[i]
                print(f"Analyzing {subj}...")
                anat, func, sbref, AP, mask, PA = self.init_func_folder(folder_name)
                denoised = self.denoise(dwi_file=AP, dwi_mask=mask)
                degibbs = self.unring(denoised)
                preprocessed = self.DWI_preproc(degibbs, PA, func)
                dwi_file = self.bias_correct(preprocessed, subj=subj)
                bias_corr_brain, bias_corr_mask = MRT_Methods.T1_correction(anat)


class rs_fMRI_prep:
    def __init__(
        self, mother_dir: str = MOTHER_DIR, subj=None, data_type: str = "func"
    ):
        self.mother_dir = mother_dir
        self.data_type = data_type
        if subj:
            subjects_dirs = [f"{mother_dir}/Niftis/{subj}"]
            subjects = [subj]
        else:
            subjects_dirs = glob.glob(f"{mother_dir}/Niftis/sub*")
            subjects = [subj.split(os.sep)[-1] for subj in subjects_dirs]
        subjects_dirs.sort()
        subjects.sort()
        self.subjects = subjects
        self.subjects_dirs = subjects_dirs

    def MotionCorrection_and_Mask(self, in_file, func_folder: str):
        mot_corrected = f"{func_folder}/{in_file.split(os.sep)[-1]}"
        if not os.path.isfile(mot_corrected):
            print("Performing Motion Correction and Brain Extraction.")
            mot_cor = Methods.MotionCorrection(in_file=in_file, out_file=mot_corrected)
            mot_corrected = mot_cor.run()
            mot_cor_mask = Methods.run_BET(mot_corrected)
            mot_cor_mask = mot_cor_mask.run()
        else:
            mot_cor_mask = mot_corrected.replace(".nii.gz", "_brain_mask.nii.gz")
        return mot_corrected, mot_cor_mask

    def init_func_folder(self, folder_name):
        func_folder = MRT_Methods.init_mrt_folder(folder_name, "func")
        anat_file, func_file, sbref_file, AP, PA = MRT_Methods.load_initial_files(
            folder_name, "func"
        )
        func_file, func_mask = self.MotionCorrection_and_Mask(func_file, func_folder)
        tmp_dir = f"{func_folder}/tmp"
        if not os.path.isdir(tmp_dir):
            os.mkdir(tmp_dir)
        files = [anat_file, sbref_file, AP, PA]
        for i, file in enumerate(files):
            files[i] = shutil.copy(file, f"{func_folder}/{file.split(os.sep)[-1]}")
        anat_file, sbref_file, AP, PA = files
        return anat_file, func_file, func_mask, sbref_file, tmp_dir, AP, PA

    def TopUp(self, AP: str, PA: str, tmp_dir: str):
        topup_field = f"{tmp_dir}/topup_res_fieldcoef.nii.gz"
        topup_res = f"{tmp_dir}/topup_nifti_res.nii.gz"
        if not os.path.isfile(topup_field):
            print(f"Performing TopUp procedure.")
            topup = Methods.TopUp(AP, PA, tmp_dir)
            topup_res, topup_field, idx_path, acq = topup.run()
        else:
            idx_path = f"{tmp_dir}/index.txt"
            acq = f"{tmp_dir}/acqparams.txt"

        return topup_field, topup_res, idx_path, acq

    def extract_mag_brain(self, field_mag: str):
        bet = Methods.mag_brain_extract(field_mag)
        mag_brain = bet.run()
        return mag_brain

    def run(self):
        for folder_name in self.subjects_dirs:
            anat, func, mask, sbref, tmp_dir, AP, PA = self.init_func_folder(
                folder_name
            )
            field_coef, field_mag, idx_path, acq = self.TopUp(AP, PA, tmp_dir)
            mag_brain = self.extract_mag_brain(field_mag)


class rs_fMRI_prep_new:
    def __init__(self, mother_dir: str = MOTHER_DIR, subj=None):
        self.mother_dir = mother_dir
        if subj:
            subjects_dirs = [f"{mother_dir}/Niftis/{subj}"]
            subjects = [subj]
        else:
            subjects_dirs = glob.glob(f"{mother_dir}/Niftis/sub*")
            subjects = [subj.split(os.sep)[-1] for subj in subjects_dirs]
        subjects_dirs.sort()
        subjects.sort()
        self.subjects = subjects
        self.subjects_dirs = subjects_dirs
        self.nifti_dir = f"{mother_dir}/Niftis"
        self.derivatives = f"{mother_dir}/Derivatives"
        self.tmp_dir = f"{mother_dir}/Derivatives/tmp"
        self.fs_license = f"{mother_dir}/license.txt"

    def init_cmd(self, nifti_dir: str, derivatives: str, subj: str):
        cmd = f"fmriprep-docker {nifti_dir} {derivatives} participant --participant_label {subj} --use-aroma --write-graph --work-dir {self.tmp_dir} --fs-licens {self.fs_license}"
        if not os.path.isdir(f"{derivatives}/fmriprep/{subj}"):
            print("Performing fmriprep preprocessing procedures")
            print(cmd)
            os.system(cmd)

    def run(self):
        for subj in self.subjects:
            print(f"Analyzing {subj}")
            self.init_cmd(self.nifti_dir, self.derivatives, subj)


# to run ICA-AROMA:
# 1. bet T1
# 2. epi-reg from rest to t1 (.mat)
# 3. fnirt from t1 to mni (.nii.gz)
# 4. mcflirt rest (.par)
