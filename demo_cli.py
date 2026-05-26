#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# CLI demo: runs MASt3R reconstruction without gradio frontend.
#   python mast3r/demo_cli.py --images photos --model_name MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric --output scene.glb
# --------------------------------------------------------
import argparse
import os
import sys
import tempfile

import torch

from mast3r.demo import get_reconstructed_scene
from mast3r.model import AsymmetricMASt3R
from mast3r.utils.misc import hash_md5

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.demo import set_print_with_timestamp

torch.backends.cuda.matmul.allow_tf32 = True


def get_args_parser():
    parser = argparse.ArgumentParser(prog='mast3r demo_cli')

    parser.add_argument('--images', type=str, default='photos',
                        help='directory containing input images (default: photos)')
    parser.add_argument('--output', type=str, default='scene.glb',
                        help='output GLB file path (default: scene.glb)')

    parser_weights = parser.add_mutually_exclusive_group(required=True)
    parser_weights.add_argument('--weights', type=str, default=None,
                                help='path to the model weights')
    parser_weights.add_argument('--model_name', type=str, default=None,
                                choices=['MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric'],
                                help='name of the model to use (from HuggingFace)')

    parser.add_argument('--device', type=str, default='cuda',
                        help='pytorch device (default: cuda)')
    parser.add_argument('--image_size', type=int, default=512,
                        choices=[512, 224], help='image size (default: 512)')
    parser.add_argument('--tmp_dir', type=str, default=None,
                        help='temporary/cache directory')

    # Reconstruction parameters (defaults matching the gradio UI)
    parser.add_argument('--optim_level', type=str, default='refine+depth',
                        choices=['coarse', 'refine', 'refine+depth'],
                        help='optimization level (default: refine+depth)')
    parser.add_argument('--lr1', type=float, default=0.07,
                        help='coarse LR (default: 0.07)')
    parser.add_argument('--niter1', type=int, default=300,
                        help='coarse iterations (default: 300)')
    parser.add_argument('--lr2', type=float, default=0.01,
                        help='fine LR (default: 0.01)')
    parser.add_argument('--niter2', type=int, default=300,
                        help='fine iterations (default: 300)')
    parser.add_argument('--min_conf_thr', type=float, default=1.5,
                        help='min confidence threshold (default: 1.5)')
    parser.add_argument('--matching_conf_thr', type=float, default=0.,
                        help='matching confidence threshold (default: 0)')
    parser.add_argument('--as_pointcloud', action='store_true', default=True,
                        help='export as pointcloud (default: True)')
    parser.add_argument('--no_pointcloud', action='store_false', dest='as_pointcloud',
                        help='export as mesh instead of pointcloud')
    parser.add_argument('--mask_sky', action='store_true', default=False,
                        help='mask sky (default: False)')
    parser.add_argument('--clean_depth', action='store_true', default=True,
                        help='clean-up depthmaps (default: True)')
    parser.add_argument('--no_clean_depth', action='store_false', dest='clean_depth',
                        help='do not clean depthmaps')
    parser.add_argument('--transparent_cams', action='store_true', default=False,
                        help='transparent cameras (default: False)')
    parser.add_argument('--cam_size', type=float, default=0.2,
                        help='camera size in output (default: 0.2)')
    parser.add_argument('--scenegraph_type', type=str, default='complete',
                        choices=['complete', 'swin', 'logwin', 'oneref'],
                        help='scenegraph type (default: complete)')
    parser.add_argument('--winsize', type=int, default=1,
                        help='window size for swin/logwin/retrieval scenegraphs')
    parser.add_argument('--win_cyclic', action='store_true', default=False,
                        help='cyclic sequence for swin/logwin (default: False)')
    parser.add_argument('--refid', type=int, default=0,
                        help='reference image id for oneref scenegraph')
    parser.add_argument('--TSDF_thresh', type=float, default=0,
                        help='TSDF threshold, >0 to enable (default: 0)')
    parser.add_argument('--shared_intrinsics', action='store_true', default=False,
                        help='optimize a single set of intrinsics (default: False)')
    parser.add_argument('--retrieval_model', type=str, default=None,
                        help='retrieval model for retrieval-based scenegraph')

    parser.add_argument('--silent', action='store_true', default=False)
    return parser


def main():
    parser = get_args_parser()
    args = parser.parse_args()
    set_print_with_timestamp()

    # Resolve weights path
    if args.weights is not None:
        weights_path = args.weights
    else:
        weights_path = "naver/" + args.model_name

    # Resolve images directory
    image_dir = args.images
    if not os.path.isdir(image_dir):
        print(f"Error: --images directory '{image_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # Gather image file paths
    supported_exts = ('.jpg', '.jpeg', '.png')
    filelist = sorted([
        os.path.join(image_dir, f) for f in os.listdir(image_dir)
        if f.lower().endswith(supported_exts)
    ])
    if len(filelist) == 0:
        print(f"Error: no images found in '{image_dir}' (supported: {supported_exts})", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(filelist)} image(s) in '{image_dir}'")

    # Load model
    print(f"Loading model from {weights_path} ...")
    model = AsymmetricMASt3R.from_pretrained(weights_path).to(args.device)
    chkpt_tag = hash_md5(weights_path)

    # Setup cache directory
    def get_context(tmp_dir):
        return tempfile.TemporaryDirectory(suffix='_mast3r_cli') if tmp_dir is None \
            else __import__('contextlib').nullcontext(tmp_dir)

    with get_context(args.tmp_dir) as tmpdirname:
        cache_dir = os.path.join(tmpdirname, chkpt_tag)
        os.makedirs(cache_dir, exist_ok=True)

        # Hardcoded output temp path (get_reconstructed_scene writes temp file, then we copy)
        outfile_name = os.path.join(tmpdirname, 'temp_scene.glb')

        # Create a dummy scene state with pre-set outfile_name to avoid tempfile.mktemp
        dummy_state = type('DummyState', (), {
            'should_delete': False,
            'cache_dir': cache_dir,
            'outfile_name': outfile_name,
        })()

        print("Running reconstruction...")
        scene_state, _ = get_reconstructed_scene(
            cache_dir,
            gradio_delete_cache=False,
            model=model,
            retrieval_model=args.retrieval_model,
            device=args.device,
            silent=args.silent,
            image_size=args.image_size,
            current_scene_state=dummy_state,
            filelist=filelist,
            optim_level=args.optim_level,
            lr1=args.lr1,
            niter1=args.niter1,
            lr2=args.lr2,
            niter2=args.niter2,
            min_conf_thr=args.min_conf_thr,
            matching_conf_thr=args.matching_conf_thr,
            as_pointcloud=args.as_pointcloud,
            mask_sky=args.mask_sky,
            clean_depth=args.clean_depth,
            transparent_cams=args.transparent_cams,
            cam_size=args.cam_size,
            scenegraph_type=args.scenegraph_type,
            winsize=args.winsize,
            win_cyclic=args.win_cyclic,
            refid=args.refid,
            TSDF_thresh=args.TSDF_thresh,
            shared_intrinsics=args.shared_intrinsics,
        )

        # Copy output to requested path
        import shutil
        output_path = os.path.abspath(args.output)
        shutil.copy2(outfile_name, output_path)
        print(f"3D model saved to {output_path}")


if __name__ == '__main__':
    main()
