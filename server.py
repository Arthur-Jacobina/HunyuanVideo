from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler
from hyvideo.constants import NEGATIVE_PROMPT
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn
from pathlib import Path
import os
from datetime import datetime
import time

def initialize_model(model_path):
    args = parse_args()
    models_root_path = Path(model_path)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")
    
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    return hunyuan_video_sampler

def generate_video(
    model,
    prompt,
    resolution,
    video_length,
    seed,
    num_inference_steps,
    guidance_scale,
    flow_shift,
    embedded_guidance_scale
):
    seed = None if seed == -1 else seed
    width, height = resolution.split("x")
    width, height = int(width), int(height)
    negative_prompt = "" # not applicable in the inference

    outputs = model.predict(
        prompt=prompt,
        height=height,
        width=width, 
        video_length=video_length,
        seed=seed,
        negative_prompt=negative_prompt,
        infer_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_videos_per_prompt=1,
        flow_shift=flow_shift,
        batch_size=1,
        embedded_guidance_scale=embedded_guidance_scale
    )
    
    samples = outputs['samples']
    
    # Save samples following the pattern from sample_video.py
    if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
        sample = samples[0].unsqueeze(0)
        save_path = os.path.join(os.getcwd(), "gradio_outputs")
        os.makedirs(save_path, exist_ok=True)
        
        time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
        video_path = f"{save_path}/{time_flag}_seed{outputs['seeds'][0]}_{outputs['prompts'][0][:100].replace('/','')}.mp4"
        save_videos_grid(sample, video_path, fps=24)
        logger.info(f'Sample saved to: {video_path}')
        
        return video_path

class VideoRequest(BaseModel):
    prompt: Optional[str] = "Extremely anxious anime girl with messy pink hair, signature thick uneven bangs covering forehead, tiny side ponytails. Large round purple eyes with visible eye bags underneath. Navy school uniform with rumpled appearance. Perpetually worried expression with comically exaggerated sweat drops."
    resolution: Optional[str] = "720x1280"
    video_length: Optional[int] = 261
    seed: Optional[int] = -1
    num_inference_steps: Optional[int] = 50
    guidance_scale: Optional[float] = 1.0
    flow_shift: Optional[float] = 7.0
    embedded_guidance_scale: Optional[float] = 6.0

app = FastAPI()
model = None

@app.on_event("startup")
async def startup_event():
    global model
    args = parse_args()
    model = initialize_model(args.model_base)

@app.post("/inference")
async def inference(request: VideoRequest):
    try:
        video_path = generate_video(
            model=model,
            prompt=request.prompt,
            resolution=request.resolution,
            video_length=request.video_length,
            seed=request.seed,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            flow_shift=request.flow_shift,
            embedded_guidance_scale=request.embedded_guidance_scale
        )
        return {"status": "success", "video_path": video_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    server_name = os.getenv("SERVER_NAME", "0.0.0.0")
    server_port = int(os.getenv("SERVER_PORT", "8081"))
    args = parse_args()
    print(args)
    uvicorn.run(app, host=server_name, port=server_port)