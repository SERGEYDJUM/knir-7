from data_generation.dro_generator import generate_phantom
import gecatsim as xc
import gecatsim.reconstruction.pyfiles.recon as recon
from gecatsim.pyfiles.CommonTools import my_path
from os import path, makedirs
import numpy as np
from PIL import Image


def create() -> None:
    size = 512
    _, mask = generate_phantom("./cfg/Phantom_Generation.json")

    mask = np.transpose(mask, (2, 0, 1)).astype(np.int8)

    x, y = np.mgrid[:size, :size]
    allowed_circle = ((x - size // 2) ** 2 + (y - size // 2) ** 2) < (size // 2) ** 2

    inv_mask = 1 - mask
    mask *= allowed_circle
    inv_mask *= allowed_circle

    with open("./cfg/dro_phantom_mask.raw", "wb") as dro_raw:
        dro_raw.write(np.ascontiguousarray(mask))

    with open("./cfg/dro_phantom_water_mask.raw", "wb") as dro_raw:
        dro_raw.write(np.ascontiguousarray(inv_mask))


def simulate() -> list[int]:
    my_path.add_search_path(path.abspath("./cfg"))
    makedirs(".temp", exist_ok=True)

    ct = xc.CatSim(
        "./cfg/Phantom_Sample.cfg",
        "./cfg/Scanner_Sample_generic.cfg",
        "./cfg/Protocol_Sample_axial.cfg",
        "./cfg/Recon.cfg",
    )  # initialization

    ##--------- Make changes to parameters (optional)
    ct.resultsName = ".temp/test"
    ct.physics.enableQuantumNoise = 1
    ct.physics.enableElectronicNoise = 1
    # ct.recon.reconType = "helical_equiAngle"

    ct.run_all()
    ct.do_Recon = 1
    recon.recon(ct)

    return [ct.recon.imageSize, ct.recon.sliceCount]


def main():
    # print(raw.min(), raw.max())
    # print(mask.min(), mask.max())

    # Image.fromarray((mask[50, :, :].astype(np.uint8) * 255)).save("mask_50.png")

    create()
    imsize, slice_cnt = simulate()

    scan = xc.rawread(".temp/test_512x512x10.raw", [slice_cnt, imsize, imsize], "float")
    scan -= np.min(scan)
    scan /= np.max(scan)
    scan *= 255
    scan = np.array(scan, dtype=np.uint8)

    for i in range(slice_cnt):
        image = Image.fromarray(scan[i, :, :])
        image.save(f"out_reconstructed/recon_slice_{i}.png")


if __name__ == "__main__":
    main()
