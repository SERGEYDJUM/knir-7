import gecatsim as xc
import gecatsim.reconstruction.pyfiles.recon as recon
from gecatsim.pyfiles.CommonTools import my_path
from os import path, makedirs
from PIL import Image, ImageDraw
import skimage
import sys
import numpy as np

from .utils import LesionBBox, Phantom
from .dro_generator import generate_phantom


def draw_voxelmap(mask: np.ndarray):
    import voxelmap as vxm

    voxels = mask.astype(np.uint8)
    voxels = skimage.measure.block_reduce(voxels, (6, 6, 6), np.max)
    model = vxm.Model(voxels)
    model.draw()


def create_phantom(show_vxm: bool = False, save_mask: bool = False) -> Phantom:
    mask, bboxes = generate_phantom("./cfg/Phantom_Generation.json")

    if show_vxm:
        draw_voxelmap(mask)

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

        img.save(f".temp/mask_{z_slices // 2}.png")

    x, y = np.mgrid[:size, :size]
    allowed_circle = ((x - size // 2) ** 2 + (y - size // 2) ** 2) < (size // 2) ** 2

    inv_mask = 1 - mask
    mask *= allowed_circle
    inv_mask *= allowed_circle

    with open("./cfg/dro_phantom_mask.raw", "wb") as dro_raw:
        dro_raw.write(np.ascontiguousarray(mask))

    with open("./cfg/dro_phantom_water_mask.raw", "wb") as dro_raw:
        dro_raw.write(np.ascontiguousarray(inv_mask))

    return Phantom(bboxes)


def simulate() -> tuple[int, int, float]:
    my_path.add_search_path(path.abspath("./cfg"))
    makedirs(".temp", exist_ok=True)

    ct = xc.CatSim(
        "./cfg/Phantom_Sample.cfg",
        "./cfg/Scanner_Sample_generic.cfg",
        "./cfg/Protocol_Sample_axial.cfg",
        "./cfg/Recon.cfg",
    )

    ct.resultsName = ".temp/test"
    ct.physics.enableQuantumNoise = 1
    ct.physics.enableElectronicNoise = 1

    scale = ct.phantom.scale * (ct.recon.imageSize**2 / (512 * ct.recon.fov))

    ct.run_all()
    ct.do_Recon = 1
    recon.recon(ct)

    return (ct.recon.imageSize, ct.recon.sliceCount, scale)


def main():
    bboxes: list[LesionBBox] = []

    if "create" in sys.argv or "demo" in sys.argv:
        bboxes = create_phantom(
            show_vxm="vxm" in sys.argv, save_mask="mask" in sys.argv
        ).signals

    imsize, slice_cnt, imscale = simulate()

    tomogram = xc.rawread(
        f".temp/test_{imsize}x{imsize}x{slice_cnt}.raw",
        [slice_cnt, imsize, imsize],
        "float",
    )

    x, y = np.mgrid[:512, :512]
    outof_circle = ((x - 512 // 2) ** 2 + (y - 512 // 2) ** 2) < (500 // 2) ** 2

    tomo_center = tomogram * outof_circle
    print(np.min(tomo_center), np.max(tomo_center))

    tomogram = np.clip(tomogram, -1000, 1000)
    tomogram += 1000
    tomogram /= 2000
    tomogram *= 255
    tomogram = np.array(tomogram, dtype=np.uint8)

    for i in range(slice_cnt):
        image = Image.fromarray(tomogram[i, :, :]).convert("RGB")
        if "demo" in sys.argv:
            draw = ImageDraw.Draw(image)
            for bbox in bboxes:
                draw.rectangle(bbox.scale(imscale).bbox(), outline=(0, 255, 0))
                draw.rectangle(bbox.scale(imscale).roi_bbox(), outline=(0, 0, 0))
        image.save(f"out_reconstructed/recon_slice_{i}.png")


if __name__ == "__main__":
    main()
