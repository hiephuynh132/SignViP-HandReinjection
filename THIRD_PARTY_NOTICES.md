# Attribution and scope

This is a curated source snapshot from the SignGen/SignViP research workspace.
Preserve copyright and license headers in all copied files.

- The original SignGen checkout's MIT license is retained in `licenses/SignGen-MIT.txt`.
- The generation code builds on SignViP and includes AnimateDiff/Diffusers-derived
  modules. Several files carry their own Apache-2.0 and other notices.
- DWPose, HaMeR, Grounded-SAM2, SAM2 and GroundingDINO have separate upstream
  packages, model licenses and, where applicable, dataset/model access terms.
  Available local license texts are preserved in `licenses/`.
- Stable Diffusion v1.5, CLIP, DINOv2, LPIPS/AlexNet and MANO assets are external.
- The license copied from the original workspace does not relicense third-party
  code, model weights, datasets or published video frames.

The paper's reported task is frame-level condition-to-video generation, using
conditions extracted from ground-truth video. This release does not claim a
complete text-to-video or temporally trained Stage II result.
