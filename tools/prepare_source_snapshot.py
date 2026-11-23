#!/usr/bin/env python3
"""Copy the paper source allowlist; never copy weights, datasets, or secrets."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import yaml
from portable_paths import sanitize_release


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve()
    target = Path(__file__).resolve().parents[1]
    if source == target:
        raise SystemExit('Source must be the original SignGen checkout.')
    copied = []

    def copy(relative, destination=None):
        src = source / relative
        dst = target / (destination or relative)
        if not src.is_file():
            raise FileNotFoundError(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append({'source': relative, 'destination': str(dst.relative_to(target)),
                       'bytes': src.stat().st_size,
                       'sha256': hashlib.sha256(src.read_bytes()).hexdigest()})

    for folder in ['signvip/models']:
        for p in sorted((source / folder).rglob('*.py')):
            copy(str(p.relative_to(source)))
    files = [
        'signvip/train_stage1_multi_cond.py', 'signvip/train_vae_decoder_reconstruction.py',
        'signvip/eval_C5_1_full_test.py', 'signvip/perceptual_losses.py', 'signvip/utils.py',
        'signvip/export_vae_training_state.py',
        'signvip/pipelines/pipeline_static.py', 'signvip/pipelines/context.py',
        'signvip/signdatasets/__init__.py', 'signvip/signdatasets/hdf5_sign_lang_video.py',
        'signvip/metrics/calculate_ssim.py',
        'signvip/make_paper_figure4_c5_comparison.py',
        'signvip/make_C5_middle_hand_comparison.py',
        'signvip/make_C5_1_vs_C5_16_hand_face_visual.py',
        'autoencoder_used/eval_autoencoders_used.py',
        'autoencoder_used/visualize_sd15_vae_middle_hand_face.py',
    ]
    for name in files:
        copy(name)
    copy('signvip/requirements.txt', 'requirements.upstream.txt')
    for original, dest in [
        ('LICENSE', 'licenses/SignGen-MIT.txt'),
        ('animatediff/LICENSE.txt', 'licenses/AnimateDiff-LICENSE.txt'),
        ('DWPose/LICENSE', 'licenses/DWPose-LICENSE.txt'),
        ('hamer/LICENSE.md', 'licenses/HaMeR-LICENSE.md'),
        ('Grounded-SAM-2/LICENSE', 'licenses/Grounded-SAM2-LICENSE.txt'),
        ('Grounded-SAM-2/LICENSE_sam2', 'licenses/SAM2-LICENSE.txt'),
        ('Grounded-SAM-2/LICENSE_groundingdino', 'licenses/GroundingDINO-LICENSE.txt'),
    ]:
        copy(original, dest)
    preprocessing = [
        'sign_language_convert.py', 'dwpose_keypoints.py', 'datasets/h5.py',
        'build_dwpose_keypoint_cache.py', 'data_center_crop.sh',
        'Grounded-SAM-2/grounded_sam2_local_demo.py',
        'hamer/generate_signvip_conditions.py', 'hamer/verify_signvip_conditions.py',
    ]
    for name in preprocessing:
        copy(name, 'preprocessing/' + name)

    # Archive the full inherited configuration chain, rather than dropping bases.
    seen = set()
    def config_chain(name):
        if name in seen:
            return
        seen.add(name)
        src = source / 'signvip/configs/stage1' / name
        cfg = yaml.safe_load(src.read_text())
        if cfg.get('base_config'):
            config_chain(cfg['base_config'])
        copy(str(src.relative_to(source)), 'provenance/original_configs/' + name)
    for name in [
        'C5.15_final_stage1_A0_best_vae.yaml',
        'C5.13_vae_reconstruction_hand4x_face4x_clip_dino_lpips.yaml',
        'C5.1_stage1_hdf5.yaml',
    ]:
        config_chain(name)

    run = 'signvip/workspace_c5_15/C5.15_final_stage1_A0_best_vae/20260920-1230'
    vae = 'signvip/workspace_resume/C5.13_vae_reconstruction_hand4x_face4x_clip_dino_lpips/20260903-1245'
    copy(run + '/config.yaml', 'provenance/paper_run_config.yaml')
    copy(run + '/best/metrics.json', 'results/proposed_best_validation.json')
    copy(vae + '/best/metrics.json', 'results/vae_best_validation.json')
    copy('signvip/reports/C5.15_full_test_chunk5_matched_baseline/metrics_summary.json',
         'results/proposed_full_test_metrics.json')
    for fig in [
        'ICEEIS_2026-proposal-method-fixed.png', 'ICEEIS_2026-Hand-Face-Mask-fixed.png',
        'sd15_vae_pretrained_128_grid.png', 'optimized_vae_reconstruction_grid.png',
        'figure4_qualitative_492_617_matched_1col_en.png',
    ]:
        copy('reports/IEEE-conference-template-062824/figures/' + fig, 'assets/' + fig)

    # Portable configs change paths only. Runtime controls are explicit below.
    cfg = yaml.safe_load((source / run / 'config.yaml').read_text())
    sd = '${oc.env:SD15_ROOT,../weights/stable-diffusion-v1-5}'
    init = '${oc.env:PAPER_INIT_ROOT,../weights/stage1_init}'
    cfg['modules'].update({
        'vae': '${oc.env:PAPER_VAE_ROOT,../weights/optimized_vae}/vae',
        'unet_2d': sd + '/unet', 'scheduler': sd + '/scheduler',
        'tokenizer': sd + '/tokenizer', 'text_encoder': sd + '/text_encoder',
        'apperance_encoder': init + '/appearance_encoder',
        'condition_encoder': init + '/condition_encoder/model.bin',
    })
    cfg['dataset']['roots'] = ['${oc.env:DATA_ROOT,../data/rwth}/train']
    cfg['dataset']['hamer_roots'] = ['${oc.env:HAMER_ROOT,../data/hamer}/train']
    cfg['dataset']['detail_mask_cache'] = '${oc.env:MASK_ROOT,../data/detail_masks}'
    cfg['validation_data']['root'] = '${oc.env:DATA_ROOT,../data/rwth}/dev'
    cfg['validation_data']['hamer_root'] = '${oc.env:HAMER_ROOT,../data/hamer}/dev'
    cfg['resume_from_checkpoint'] = None
    cfg.pop('config', None)
    cfg.pop('exp_name', None)
    # Match the saved selected checkpoint's recorded selection rule.
    cfg['best_checkpoint'] = {'metric': 'global_ssim'}
    cfg['checkpointing_steps'] = 1000
    cfg['max_ckpt'] = 1
    cfg['max_model'] = 1
    out = target / 'signvip/configs/paper'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'proposal.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))

    vcfg = yaml.safe_load((source / 'signvip/configs/stage1/C5.13_vae_reconstruction_hand4x_face4x_clip_dino_lpips.yaml').read_text())
    vcfg['pretrained_vae'] = sd + '/vae'
    vcfg['workspace_root'] = '${oc.env:SIGNVIP_WORKSPACE_ROOT,../outputs/vae}'
    vcfg['dataset'].update(train_root='${oc.env:DATA_ROOT,../data/rwth}/train',
                           dev_root='${oc.env:DATA_ROOT,../data/rwth}/dev',
                           detail_mask_cache='${oc.env:MASK_ROOT,../data/detail_masks}')
    vcfg['solver']['max_train_steps'] = 200000
    vcfg['perceptual_loss']['clip_model'] = '${oc.env:CLIP_MODEL,openai/clip-vit-large-patch14}'
    vcfg['perceptual_loss']['dino_model'] = '${oc.env:DINO_MODEL,facebook/dinov2-small}'
    (out / 'vae.yaml').write_text(yaml.safe_dump(vcfg, sort_keys=False))
    (target / 'provenance/source_manifest.json').write_text(json.dumps(copied, indent=2) + '\n')
    sanitize_release(target, source)
    print(f'Copied {len(copied)} source/config/figure files; no weights or datasets copied.')


if __name__ == '__main__':
    main()
