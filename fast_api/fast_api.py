from pydantic import BaseModel
from fastapi import FastAPI, Response
import json
import torch
from model.model_definition import Variational_Autoencoder
from model.model import get_image_by_id, model_interp, random_player


MODEL_VERSION = "0.0.1" # Could be an env var or param


print('Loading app...')
app = FastAPI()


print(f'Loading model version {MODEL_VERSION}...')
saved_model = Variational_Autoencoder(500)
saved_model.load_state_dict(torch.load(f'vae700.pth', map_location="cpu"))
saved_model.eval()
print('Model ready...')


@app.get("/image/{image_index}")
def get_image(image_index):
    im = get_image_by_id(image_index).permute(1,2,0)
    return {
        "image": json.dumps(im.tolist())
    }

@app.get("/random")
def get_image():
    im = random_player(saved_model).permute(1,2,0)
    return {
        "image": json.dumps(im.tolist())
    }


class InterpolationRequest(BaseModel):
    Image1: str = 0
    Image2: int = 1
    size: int = 100


@app.post("/interpolation")
def predict(request_body: InterpolationRequest, response: Response):

    interp_result = model_interp(model = saved_model, index1 = request_body.Image1, index2 = request_body.Image2, size = request_body.size)

    interp_result = interp_result.permute(0,2,3,1)

    return {
        "images": json.dumps(interp_result.tolist())
    }

