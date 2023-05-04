import modules.scripts as scripts
import gradio as gr
from fastapi import FastAPI, Body
from modules.processing import Processed, StableDiffusionProcessingImg2Img, process_images, images, fix_seed

from modules.utils import import_or_install
from modules.processing import Processed
from modules.shared import opts, cmd_opts, state
import numpy as np
import cv2
from PIL import Image
from rembg import remove, new_session

class Script(scripts.Script):

    def title(self):
        return "Studio Global Change Background"

    def run(self, p, annotator_resolution,canny_low_threshold,canny_high_threshold):
        if image is None:
            image=p.init_images[0]
        only_mask=True
        import_or_install("rembg","rembg[gpu]")
        session=new_session()
        mask=remove(image,only_mask=True,session=session)
        p.image_mask=mask
        proc = process_images(p)
        proc.images.append(mask)
        return proc
    
def clare_api(_: gr.Blocks, app: FastAPI):
    @app.get("/studioglobal/status")
    async def get_status():
        return {"status": "ok", "version": "1.0.0"}
    @app.post("/studioglobal/change_background")
    async def post_change_background(image_str: str = Body(...)):
        import base64
        import io
        image_bytes = base64.b64decode(image_str)
        image = Image.open(io.BytesIO(image_bytes),formats=["PNG"])