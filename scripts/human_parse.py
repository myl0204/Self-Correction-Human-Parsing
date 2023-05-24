import sys

from modules import scripts, processing, shared
from simple_extractor import parse_image
from PIL import ImageOps, ImageChops


class HumanParseScript(scripts.Script):

    def __init__(self) -> None:
        super().__init__()

    def title(self):
        return "HumanParse"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def check_script_enable(self, p: processing.StableDiffusionProcessing):

        try:
            script_runner = p.scripts
            if script_runner is None:
                return False

            human_parse_script = None
            for script in script_runner.alwayson_scripts:
                if script.title() == 'HumanParse':
                    human_parse_script = script

            if human_parse_script is None:
                return False

            script_args = p.script_args[human_parse_script.args_from:human_parse_script.args_to]

            args = script_args[0]
            print(f"check_script_enable enable args, {args}", file=sys.stderr)
            return args

        except Exception as e:
            print(e, file=sys.stderr)
            return False

    def process(self, p, *args):
        enable = self.check_script_enable(p)
        if enable:
            print("check_script_enable result true", file=sys.stderr)

        if not enable:
            print("check_script_enable result false", file=sys.stderr)

        # get orig image
        image = getattr(p, "init_images", [None])[0]

        # parse image
        mask_image = parse_image(image, shared.opts.outdir_img2img_samples)

        # inpaint
        alpha_mask = ImageOps.invert(image.split()[-1]).convert('L').point(lambda x: 255 if x > 0 else 0, mode='1')
        mask_image = ImageChops.lighter(alpha_mask, mask_image.convert('L')).convert('L')

        # set mask image for process
        p.image_mask = mask_image
        return


