from _init_paths import *
from libs.utils import *
from libs.utils.commons import *
from libs.wrappers.xmem_wrapper import XMemWrapper


def run_xmem_segmentation(rgb_images, mask_0, save_folder):
    xmem_wrapper = XMemWrapper(device="cuda")
    xmem_wrapper.reset()

    for i, rgb_image in enumerate(rgb_images):
        if i == 0:
            mask = xmem_wrapper.get_mask(rgb_image, mask_0)
        else:
            mask = xmem_wrapper.get_mask(rgb_image)

        write_mask_image(Path.joinpath(save_folder, f"mask_{i:06d}.png"), mask)

        # draw mask over image
        vis = np.zeros_like(rgb_images[0])
        vis[mask > 0] = 255
        vis = cv2.addWeighted(rgb_image, 0.5, vis, 0.5, 0)
        write_rgb_image(Path.joinpath(save_folder, f"vis_{i:06d}.png"), vis)


if __name__ == "__main__":
    rgb_files = sorted(Path(PROJ_ROOT / "demo" / "color_images").glob("*.jpg"))
    rgb_images = [read_rgb_image(f) for f in rgb_files]
    mask_0 = read_mask_image(Path(PROJ_ROOT / "demo" / "mask_000000.png"))

    save_folder = Path(PROJ_ROOT / "demo/segmentation/xmem_segmentation")
    save_folder.mkdir(exist_ok=True, parents=True)

    run_xmem_segmentation(rgb_images, mask_0, save_folder)
