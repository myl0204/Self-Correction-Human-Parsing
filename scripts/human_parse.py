import sys
import gc

from modules import scripts, processing, shared, devices
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
            print(f"check_script_enable enable args, {args}")
            return args

        except Exception as e:
            print(e)
            return False

    def process(self, p, *args):
        enable = self.check_script_enable(p)
        if enable:
            print("check_script_enable result true")

        if not enable:
            print("check_script_enable result false")

        # get orig image
        image = getattr(p, "init_images", [None])[0]

        # parse image
        mask_image = parse_image(image, shared.opts.outdir_img2img_samples)

        # set mask image for process
        p.image_mask = mask_image
        return

    def postprocess(self, p, processed, *args):
        gc.collect()
        devices.torch_gc()


