# %%
import os
import glob
import json
import pandas as pd

proj_dir = r"/home/gal/Brain_Networks"
TEMPLATE_PARTICIPANTS = r"/media/gal/New Volume/FSL_pipeline/bids-starter-kit-master/templates/participants.tsv"
MRICRO_PATH = r"/home/gal/Utillitis/MRIcroGL/Resources"
ds_description = r"/home/gal/dataset_description.json"
CRF_FILE = os.path.abspath("/home/gal/CRF.xlsx")
os.path.isfile(CRF_FILE)
df = pd.read_excel(CRF_FILE, header=0)

# %%


class Organize_by_BIDS:
    def __init__(
        self,
        proj_dir: str = proj_dir,
        participants_temp: str = TEMPLATE_PARTICIPANTS,
        mricro_path: str = MRICRO_PATH,
        ds_description: str = ds_description,
        df: pd.DataFrame = df,
    ):
        self.proj_dir = proj_dir
        self.participants_temp = participants_temp
        self.mricro_path = mricro_path
        self.ds_description = ds_description
        self.df = df
        self.main_branch()

    #%%print('Building main branch of the project...')

    def main_branch(self):
        print("Building main branch of the project...")
        branches = ["Niftis", "Derivatives"]
        for branch in branches:
            if not os.path.isdir(f"{self.proj_dir}/{branch}"):
                os.makedirs(f"{self.proj_dir}/{branch}")

    def set_directories(self, subnum):
        newdir = f"{self.proj_dir}/Niftis/{subnum}/anat"
        if not os.path.isdir(newdir):
            os.makedirs(newdir)
            os.makedirs(newdir.replace("anat", "dwi"))
            os.makedirs(newdir.replace("anat", "func"))

    def participants(
        self,
        sub: str,
        age: str,
        hand: str,
        sex: str,
        participant_tsv: str,
        temp_participants: str,
    ):
        if os.path.isfile(participant_tsv) == False:
            df = pd.read_csv(temp_participants)
        else:
            df = pd.read_csv(participant_tsv, sep="\t")
        loc = int(sub[-2:])
        newline = [sub, age, hand, sex]
        df.loc[loc - 1] = newline
        df.to_csv(participant_tsv, sep="\t", index=False)
        print("Created participants.tsv")

    #%%
    def dataset_description(self, ds_description: str):
        Proj_Name = (
            "Networks_Dynamics"  # Define project name as it will be in the .json file
        )
        replacements = {"proj_name": Proj_Name}
        #        if os.path.isfile(
        #            "{0}/dataset_description.json".format(os.path.dirname(new_toplvl))
        #        ):
        with open(ds_description) as infile:
            with open(
                "{0}/dataset_description.json".format(self.proj_dir), "w"
            ) as outfile:
                for line in infile:
                    for src, target in replacements.items():
                        line = line.replace(src, target)
                    outfile.write(line)
        print("Created dataset_description.json")

    #%%
    def make_anat(self, subnum):
        dcm = glob.glob(f"{self.proj_dir}/sourcedata/{subnum}/*MPRAGE*")[0]
        new_nii = f"{self.proj_dir}/Niftis/{subnum}/anat"
        n_subject = subnum.split("-")[-1]
        cmd = "{0}/dcm2niix -o {1} -f {2}_%f_%p {3}".format(
            self.mricro_path, new_nii, n_subject, dcm
        )
        os.system(cmd)
        self.fix_anat_hdr(subnum)

    def make_DTI(self, subnum):
        files = glob.glob(f"{self.proj_dir}/sourcedata/{subnum}/*ep2d*")
        for f in files:
            new_nii = f"{self.proj_dir}/Niftis/{subnum}/dwi"
            n_subject = subnum.split("-")[-1]
            cmd = "{0}/dcm2niix -o {1} -f {2}_%f_%p {3}".format(
                self.mricro_path, new_nii, n_subject, f
            )
            os.system(cmd)
        self.fix_DTI_hdr(subnum)

    def make_rf(self, subnum):
        files = glob.glob(f"{self.proj_dir}/sourcedata/{subnum}/*rfMRI*")
        for f in files:
            new_nii = f"{self.proj_dir}/Niftis/{subnum}/func"
            n_subject = subnum.split("-")[-1]
            cmd = "{0}/dcm2niix -o {1} -f {2}_%f_%p {3}".format(
                self.mricro_path, new_nii, n_subject, f
            )
            os.system(cmd)
        self.fix_rf_hdr(subnum)

    def run(self):
        subjects = glob.glob(f"{self.proj_dir}/sourcedata/sub-*")
        subjects.sort()
        for subj in subjects:
            subnum = subj.split(os.sep)[-1]
            self.set_directories(subnum=subnum)
            data = df.iloc[int(subnum.split("-")[-1]) - 1]
            identifier = str(data["subnum"].split("-")[-1])
            age = data.Age
            hand = data.Hand
            sex = data.Gender
            self.participants(
                sub=identifier,
                age=age,
                hand=hand,
                sex=sex,
                participant_tsv=f"{self.proj_dir}/participants.tsv",
                temp_participants=TEMPLATE_PARTICIPANTS,
            )
            self.dataset_description(ds_description=ds_description)
            self.make_anat(subnum=subnum)
            self.make_DTI(subnum=subnum)
            self.make_rf(subnum=subnum)
            self.sort_derivatives(subnum=subnum)

    def fix_anat_hdr(self, subnum):
        MPRAGE_files = glob.glob(f"{self.proj_dir}/Niftis/{subnum}/anat/*MPRAGE*")
        for file in MPRAGE_files:
            file_list = file.split(os.sep)
            MP_hdr = file_list[-1].split(".")
            MP_hdr[0] = f"{subnum}_T1w"
            T1w_hdr = ".".join(MP_hdr)
            os.rename(file, f"{self.proj_dir}/Niftis/{subnum}/anat/{T1w_hdr}")

    def fix_DTI_hdr(self, subnum):
        DTI_files = glob.glob(f"{self.proj_dir}/Niftis/{subnum}/dwi/*ep2d*")
        for file in DTI_files:
            file_hdr = file.split(os.sep)[-1]
            file_data = file_hdr.split("_")[-1]
            orient, type = file_data.split(".")[0], ".".join(file_data.split(".")[1:])
            new_hdr = f"{subnum}_acq-{orient}_dwi.{type}"
            os.rename(file, f"{os.path.dirname(file)}/{new_hdr}")

    def fix_rf_hdr(self, subnum):
        rf_files = glob.glob(f"{self.proj_dir}/Niftis/{subnum}/func/*rfMRI*")
        for file in rf_files:
            file_hdr = file.split(os.sep)[-1]
            file_data = file_hdr.split("_")[-1]
            type = ".".join(file_data.split(".")[1:])
            if "SBRef" not in file:
                new_hdr = f"{subnum}_task-rest_bold.{type}"
            else:
                new_hdr = f"{subnum}_task-rest_sbref.{type}"
            if type == "json":
                with open(file, "r+") as f:
                    data = json.load(f)
                    data["TaskName"] = "Rest"
                    f.seek(0)
                    json.dump(data, f, indent=4)
                    f.truncate()
            os.rename(file, f"{os.path.dirname(file)}/{new_hdr}")

    def sort_derivatives(self, subnum: str):
        deriv_dir = f"{self.proj_dir}/Derivatives/dwi/{subnum}"
        fmap_dir = f"{self.proj_dir}/Niftis/{subnum}/fmap"
        if not os.path.isdir(deriv_dir):
            os.makedirs(deriv_dir)
        if not os.path.isdir(fmap_dir):
            os.makedirs(fmap_dir)
        files_to_move = glob.glob(f"{self.proj_dir}/Niftis/{subnum}/dwi/*PA*")
        for f in files_to_move:
            hdr = f.split(os.sep)[-1]
            os.rename(f, f"{fmap_dir}/{hdr}")
