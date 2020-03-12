from Networks_plasticity import Preprocessing

prep = Preprocessing.rs_fMRI_prep(
    mother_dir="/home/gal/Networks_Dynamics", subj="sub-01", data_type="func"
)
prep.run()
