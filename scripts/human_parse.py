import modules.scripts as scripts
from modules.shared import opts
from simple_extractor import parse_image
from PIL import ImageOps, ImageChops


class HumanParseScript(scripts.Script):

    def __init__(self) -> None:
        super().__init__()

    def title(self):
        return "HumanParse"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def process(self, p, *args):
        # get orig image
        image = getattr(p, "init_images", [None])[0]

        # parse image
        mask_image = parse_image(image, opts.outdir_img2img_samples)

        # inpaint
        alpha_mask = ImageOps.invert(image.split()[-1]).convert('L').point(lambda x: 255 if x > 0 else 0, mode='1')
        mask_image = ImageChops.lighter(alpha_mask, mask_image.convert('L')).convert('L')

        # set mask image for process
        p.image_mask = mask_image
        return


