import modules.scripts as scripts
from modules.shared import opts
from simple_extractor import parse_image


class HumanParseScript(scripts.Script):

    def __init__(self) -> None:
        super().__init__()

    def title(self):
        return "HumanParse"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def process(self, p, *args):
        # get orig image
        orig_image = getattr(p, "init_images", [None])[0]

        # parse image
        mask_image = parse_image(orig_image, opts.outdir_img2img_samples)

        # set mask image for process
        p.image_mask = mask_image
        return


