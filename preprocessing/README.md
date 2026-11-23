# Raw data and condition preparation

The paper consumes aligned RGB/pose HDF5, rendered HaMeR HDF5, Grounded-SAM2
hand/face masks and DWPose keypoint caches. These files are external inputs,
not copied dataset contents.

The original project-specific adapter sources are preserved here:

1. `sign_language_convert.py`, `datasets/h5.py`, `dwpose_keypoints.py`,
   `data_center_crop.sh`: raw RWTH conversion and aligned center-crop inputs.
2. `hamer/generate_signvip_conditions.py`: render HaMeR conditions aligned to
   the HDF5 frames; `verify_signvip_conditions.py` checks paired data.
3. `Grounded-SAM-2/grounded_sam2_local_demo.py`: Grounded-SAM2 region extraction.
4. `build_dwpose_keypoint_cache.py`: keypoint NPZ caches for regional evaluation.

These are integration scripts, not standalone dependency bundles. The conversion
script has imports for the original project's CLIP, DWPose/ControlNet, TensorFlow
and optional DPT paths; HaMeR needs its full upstream package, ViTPose/detector
dependencies and licensed model assets. The Grounded-SAM2 script needs the
upstream SAM2/GroundingDINO code, configs and weights. Install those upstream
repositories separately and place the adapters in the corresponding original
layout, updating machine-specific paths before use. The portable training/eval
entry points instead start from prepared HDF5 and NPZ inputs.

Do not apply the thesis keypoint fusion, Sapiens preference, VSL-UTE dataset
processing or thesis dilation settings to the paper reproduction.

Expected mask filename at paper resolution:
`<text_key>_grounded_sam2_face_hands_384_to128_v4.npz`.
Expected DWPose keypoint filename:
`<text_key>_dwpose384_keypoints_v1.npz` (check the cache's inference resolution).
Frame indices, text keys and RGB/condition alignment must agree between inputs.

Original local locations and fields for permitted Drive uploads are documented
in `../WEIGHT_PATHS.txt`. Raw RWTH data access remains with the dataset provider.
