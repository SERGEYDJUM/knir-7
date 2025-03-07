import gecatsim as xc
import gecatsim.reconstruction.pyfiles.recon as recon
from gecatsim.pyfiles.CommonTools import my_path as catsim_paths

from PIL import Image, ImageDraw
from numpy.typing import NDArray
from os import path, makedirs
from argparse import ArgumentParser, Namespace
from datetime import datetime
from hashlib import md5
import numpy as np
import csv

from .utils import Phantom
from .dro_generator import generate_phantom

WORKDIR = path.abspath("./.temp")
CFGDIR = path.abspath("./cfg")
DATASET_DIR = path.abspath("./dataset")
EXPERIMENT_PREFIX = path.join(WORKDIR, "MAIN")

CSV_HEADER = (
    "slice_index",
    "bbox_index",
    "bbox_center_x",
    "bbox_center_y",
    "bbox_radius_x",
    "bbox_radius_y",
    "bbox_safe_r_x",
    "bbox_safe_r_y",
    "signal_present",
    "tube_current",
    "recon_kernel",
    "phantom_cfg_md5",
    "xcist_cfg_md5",
)


def create_phantom(save_mask: bool = False, empty: bool = False) -> Phantom:
    mask, bboxes = generate_phantom(
        path.join(CFGDIR, "Phantom_Generation.json"), empty=empty
    )

    mask = np.transpose(mask, (2, 0, 1)).astype(np.int8)
    assert mask.shape[1] == mask.shape[2]

    z_slices = mask.shape[0]
    size = mask.shape[2]

    if save_mask:
        img = Image.fromarray((mask[z_slices // 2, :, :].astype(np.uint8) * 255))
        draw = ImageDraw.Draw(img)
        for bbox in bboxes:
            draw.rectangle(bbox.bbox(), outline=192)
            draw.rectangle(bbox.roi_bbox(), outline=96)
            draw.rectangle(bbox.safe_bbox(), outline=32)

        img.save(path.join(WORKDIR, f"mask_{z_slices // 2}.png"))

    x, y = np.mgrid[:size, :size]
    allowed_circle = ((x - size // 2) ** 2 + (y - size // 2) ** 2) < (size // 2) ** 2

    inv_mask = 1 - mask
    mask *= allowed_circle
    inv_mask *= allowed_circle

    with open(path.join(CFGDIR, "dro_phantom_mask.raw"), "wb") as dro_raw:
        dro_raw.write(np.ascontiguousarray(mask))

    with open(path.join(CFGDIR, "dro_phantom_water_mask.raw"), "wb") as dro_raw:
        dro_raw.write(np.ascontiguousarray(inv_mask))

    return Phantom(bboxes)


def reconstruct(catsim: xc.CatSim) -> tuple[NDArray, float]:
    catsim.do_Recon = 1
    recon.recon(catsim)

    imsize, slice_cnt = catsim.recon.imageSize, catsim.recon.sliceCount
    bbox_scale = catsim.phantom.scale * (imsize**2 / (512 * catsim.recon.fov))

    tomogram = xc.rawread(
        EXPERIMENT_PREFIX + f"_{imsize}x{imsize}x{slice_cnt}.raw",
        [slice_cnt, imsize, imsize],
        "float",
    )

    return tomogram, bbox_scale


def demo_reconstructed(tomogram: NDArray, bboxes: Phantom, imscale: float) -> None:
    demo_dir = path.join(WORKDIR, "out")
    makedirs(demo_dir, exist_ok=True)

    demo_tomo = np.clip(tomogram, -1000, 1000)
    demo_tomo += 1000
    demo_tomo *= 255 / 2000
    demo_tomo = np.array(demo_tomo, dtype=np.uint8)

    for i in range(demo_tomo.shape[0]):
        image = Image.fromarray(demo_tomo[i, :, :]).convert("RGB")
        draw = ImageDraw.Draw(image)
        for bbox in bboxes.signals:
            draw.rectangle(bbox.scale(imscale).bbox(), outline=(0, 255, 0))
            draw.rectangle(bbox.scale(imscale).roi_bbox(), outline=(0, 0, 0))
            draw.rectangle(bbox.scale(imscale).safe_bbox(), outline=(0, 0, 255))
        image.save(path.join(demo_dir, f"recon_slice_{i}.png"))


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-d", "--demo", action="store_true")
    parser.add_argument("-m", "--mask", action="store_true")
    parser.add_argument("-s", "--skipgen", action="store_true")
    parser.add_argument("-e", "--empty", action="store_true")
    parser.add_argument("-r", "--repeat", default=1, type=int)
    return parser.parse_args()


def main():
    args = parse_args()

    catsim_paths.add_search_path(path.abspath(CFGDIR))

    bbox_cache = path.join(WORKDIR, "bboxes.json")
    catsim_cfg = path.join(CFGDIR, "CatSim.cfg")
    phantomgen_cfg = path.join(CFGDIR, "Phantom_Generation.json")

    dataset_csv = path.join(DATASET_DIR, "dataset.csv")
    dataset_cfgs = path.join(DATASET_DIR, "cfgs")
    dataset_raws = path.join(DATASET_DIR, "raws")

    makedirs(WORKDIR, exist_ok=True)
    makedirs(dataset_cfgs, exist_ok=True)
    makedirs(dataset_raws, exist_ok=True)

    # Phantom generation
    bboxes: Phantom = None
    if not args.skipgen:
        bboxes = create_phantom(save_mask=args.mask, empty=args.empty)
        bboxes.dump(bbox_cache)
    else:
        bboxes = Phantom.load(bbox_cache)

    cfg_timestamp = datetime.now().isoformat(timespec="seconds").replace(":", "")
    ph_ds_path = path.join(dataset_cfgs, f"{cfg_timestamp}.json")
    ph_cfg_hash: str = None

    with (
        open(phantomgen_cfg, "rb") as ifile,
        open(ph_ds_path, "wb") as ofile,
    ):
        contents = ifile.read()
        ph_cfg_hash = md5(contents).hexdigest()
        ofile.write(contents)

    cs_ds_path = path.join(dataset_cfgs, f"{cfg_timestamp}.cfg")
    catsim_cfg_hash: str = None

    with (
        open(catsim_cfg, "rb") as ifile,
        open(cs_ds_path, "wb") as ofile,
    ):
        contents = ifile.read()
        catsim_cfg_hash = md5(contents).hexdigest()
        ofile.write(contents)

    # Simulation
    catsim = xc.CatSim(catsim_cfg)
    catsim.resultsName = EXPERIMENT_PREFIX

    if not path.exists(dataset_csv):
        with open(dataset_csv, "w", encoding="utf-8") as csvfile:
            csvfile.write(",".join(CSV_HEADER) + "\n")

    with open(dataset_csv, "a", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, lineterminator="\n")

        for repeat_iter in range(args.repeat):
            catsim.run_all()
            tomogram, imscale = reconstruct(catsim)

            raw_timestamp = (
                datetime.now().isoformat(timespec="seconds").replace(":", "")
            )
            raw_path = path.join(dataset_raws, f"{raw_timestamp}.raw")

            with open(raw_path, "wb") as raw_file:
                raw_file.write(tomogram)

            for slice_idx in range(tomogram.shape[0]):
                for bb_idx, bbox in enumerate(bboxes.signals):
                    if abs(imscale - 1) > 1e-3:
                        bbox = bbox.scale(imscale)

                    writer.writerow(
                        (
                            slice_idx,
                            bb_idx,
                            bbox.center[0],
                            bbox.center[1],
                            bbox.r[0],
                            bbox.r[1],
                            bbox.sr[0],
                            bbox.sr[1],
                            not args.empty,
                            int(catsim.protocol.mA),
                            catsim.recon.kernelType,
                            ph_cfg_hash,
                            catsim_cfg_hash,
                        )
                    )

            if args.demo:
                demo_reconstructed(tomogram, bboxes, imscale)


if __name__ == "__main__":
    main()
