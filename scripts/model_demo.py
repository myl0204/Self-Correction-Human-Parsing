import os
import gradio as gr

from modules import script_callbacks, ui_common
from modules.shared import opts
from io import BytesIO
import base64
import requests
from datetime import datetime

from PIL import Image, PngImagePlugin, ImageOps, ImageChops

def on_ui_tabs():

    def decode_base64_to_image(image_encoded: str) -> Image.Image:
        return Image.open(BytesIO(base64.b64decode(image_encoded)))

    def encode_pil_to_base64(pil_image):
        with BytesIO() as output_bytes:

            # Copy any text-only metadata
            use_metadata = False
            metadata = PngImagePlugin.PngInfo()
            for key, value in pil_image.info.items():
                if isinstance(key, str) and isinstance(value, str):
                    metadata.add_text(key, value)
                    use_metadata = True

            pil_image.save(
                output_bytes, "PNG", pnginfo=(metadata if use_metadata else None)
            )
            bytes_data = output_bytes.getvalue()
        base64_str = str(base64.b64encode(bytes_data), "utf-8")
        return "data:image/png;base64," + base64_str

    def submit(image, use_human_parse_flag, use_canny_flag, use_openpose_flag, batch_size):
        url_img2img = "http://101.35.154.34:7860/sdapi/v1/img2img"
        simple_img2img = {
            "init_images": [],
            "resize_mode": 0,
            "denoising_strength": 1,
            "mask": None,
            "mask_blur": 4,
            "inpainting_fill": 0,
            "inpaint_full_res": 1,
            "inpaint_full_res_padding": 32,
            "inpainting_mask_invert": 0,
            "prompt": "example prompt",
            "styles": [],
            "seed": -1,
            "subseed": -1,
            "subseed_strength": 0,
            "seed_resize_from_h": -1,
            "seed_resize_from_w": -1,
            "batch_size": batch_size,
            "n_iter": 1,
            "steps": 30,
            "cfg_scale": 12,
            "width": 512,
            "height": 512,
            "restore_faces": False,
            "tiling": False,
            "negative_prompt": "",
            "eta": 0,
            "s_churn": 0,
            "s_tmax": 0,
            "s_tmin": 0,
            "s_noise": 1,
            "override_settings": {},
            "sampler_index": "DPM++ 2M Karras",
            "include_init_images": False,
            "alwayson_scripts": {
                "ControlNet": {
                    "args": [
                        {
                            "enabled": use_canny_flag,
                            "module": "canny",
                            "model": "control_sd15_canny [fef5e48e]",
                            "weight": 1.0,
                            "image": encode_pil_to_base64(image),
                            "resize_mode": 1,
                            "control_mode": 0,
                            "lowvram": False,
                            "processor_res": 512,
                            "threshold_a": 100,
                            "threshold_b": 200,
                            "pixel_perfect": False
                        },  {
                            "enabled": use_openpose_flag,
                            "module": "openpose_full",
                            "model": "control_v11p_sd15_openpose [cab727d4]",
                            "weight": 1.0,
                            "image": encode_pil_to_base64(image),
                            "resize_mode": 1,
                            "control_mode": 0,
                            "lowvram": False,
                            "processor_res": 512,
                            "guidance_start": 0.0,
                            "guidance_end": 1.0,
                            "pixel_perfect": False
                        }
                    ]
                }
                ,
                "HumanParse": {
                    "args": [{
                        "enabled": use_human_parse_flag
                    }]
                }
            }
        }

        image = image.convert("RGB")
        simple_img2img["init_images"] = [encode_pil_to_base64(image)]
        #prompt: <lora:koreanDollLikeness_v15:0.4>, (original: 1.2), (realistic: 1.3) (mixed Korean), beautiful girl with beautiful details, extremely detailed eyes and face, eyes with beautiful details, huge file size, ultra detail, high resolution, ultra detailed, best quality, ultra detailed and beautiful, illustration, ultra detailed and beautiful, CG, unity, 8k wallpaper, amazing, fine Detail, masterpiece, top quality, official art, extremely detailed, CG unity 8k wallpaper, cinematic lighting, (perfect shiny skin:0.6), (light skin:1.2),brown hair, slim and smooth lines, blushing, brown hair,  standing, happy, (smile:1.3), front view,  streets, bare neck, (bare arms:1.4), smoth hair,  slim, white color long pants
        simple_img2img["prompt"] = '<lora:koreanDollLikeness_v15:0.4>((best quality)),((masterpiece)),((realistic))(mixed Korean), (detailed), (1girl)women, (perfect shiny skin:0.6), (light skin:1.2),brown hair, slim and smooth lines, blushing, brown hair, small breast, bangs, freckles, standing, (smile:1.1),front view, streets, bare neck, (bare arms:1.4), smoth hair, slim, white color long pants, (happy:1.1),((masterpiece)), absurdres'
        #negative prompt: negetive:lowres , text, error, cropped, worst quality, low quality, normal quality, artifacts, signature, watermark, username, blurry, paintings, sketches, monochrome, grayscale, ugly, duplicate, morbid, mutilated, out of frame, bad anatomy , mutation, deformed, dehydrated,  bad proportions, disfigured, gross proportions, dot, mole, cloned face, poorly drawn face, unclear eyes, long neck, humpbacked, bad hands, mutated hands, poorly drawn hands, fused fingers, too many fingers, missing fingers, extra fingers, extra digit, fewer digits, missing arms, extra arms, bad feet, extra limbs, malformed limbs, missing legs, extra legs
        simple_img2img["negative_prompt"] = '(deformed iris, deformed pupils, bad eyes, semi-realistic:1.4)(bad-image-v2-39000,bad prompt_version2, bad hands-5, NG_DeepNegative_V1_41, bad-artist-anime:0.7), (worst quality, low quality:1.3),(depth of field, blurry:1.2), (greyscale, monochrome:1.1), nose, cropped, lowres, text, jpeg artificat, signature, watermark, username, blurry, artist name, trademark, watermark, title, (tan, muscular, loli, petite, child, infant, toddler,  chibi, sd character:1.1), multiple view, Reference sheet, long neck, nsfw'
        response = requests.post(url_img2img, json=simple_img2img)

        response_json = response.json()
        images = response_json['images']

        results = []
        now = datetime.now()
        i = 0
        while i < batch_size:
            result = decode_base64_to_image(images[i])
            image_name = 'model_demo_result_' + now.strftime("%Y-%m-%d %H:%M:%S") + '_' + str(i) + '.jpg'
            result.save(os.path.join(opts.outdir_img2img_samples, image_name))

            results.append(result)
            i = i + 1

        return results

    with gr.Blocks(analytics_enabled=False) as model_demo:
        with gr.Row():
            with gr.Column(variant='panel'):
                upload_image = gr.Image(elem_id="model_demo_image", label="原图", source="upload", interactive=True, type="pil")
                batch_size_num = gr.Slider(minimum=1, maximum=8, step=1, label='生成数量', value=1,
                                       elem_id="model_demo_batch_size")

                use_human_parse = gr.Checkbox(label='使用human parse', value=True, elem_id="model_demo_use_human_parse")
                use_canny = gr.Checkbox(label='使用ControlNet canny', value=True, elem_id="model_demo_use_canny")
                use_openpose = gr.Checkbox(label='使用ControlNet openpose', value=True, elem_id="model_demo_use_openpose")

            with gr.Column(variant='result'):
                submit_button = gr.Button('生成', elem_id="model_demo_generate", variant='primary')
                model_demo_gallery = gr.Gallery(label='结果', show_label=False, elem_id="model_demo_gallery").style(grid=4)

        submit_button.click(
            fn=submit,
            inputs=[upload_image, use_human_parse, use_canny, use_openpose, batch_size_num],
            outputs=[
                model_demo_gallery
            ]
        )

    return [(model_demo, "模特换装", "model_demo")]


script_callbacks.on_ui_tabs(on_ui_tabs)