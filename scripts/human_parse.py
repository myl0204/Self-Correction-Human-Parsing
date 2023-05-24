import gc
import gradio as gr

from modules import scripts, processing, shared, devices
from simple_extractor import parse_image
from modules.ui_components import FormRow


class HumanParseScript(scripts.Script):

    def __init__(self) -> None:
        super().__init__()

    def title(self):
        return "HumanParse"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with FormRow(elem_classes="checkboxes-row", variant="compact"):
            enabled = gr.Checkbox(label='Enable', value=False, visible=False)

        return [enabled]

    # 校验脚本是否启用
    def check_script_enable(self, p: processing.StableDiffusionProcessing):

        try:
            human_parse_script = None
            for script in p.scripts.alwayson_scripts:
                if script.title() == 'HumanParse':
                    human_parse_script = script

            script_args = p.script_args[human_parse_script.args_from:human_parse_script.args_to]
            if script_args is not None and len(script_args) > 0:
                args = script_args[0]
                return args['enabled']

            return False

        except Exception as e:
            print(e)
            return False

    def process(self, p, *args):
        enable = self.check_script_enable(p)
        print(f"human parse check_script_enable result, {enable}")
        if not enable:
            return

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


