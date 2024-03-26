from diffusers import AutoPipelineForText2Image
import torch

num_of_gpus = torch.cuda.device_count()
print("Found %d GPUs" % num_of_gpus)

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16", local_files_only=True)
pipe.to("cuda")

steps = 1
image = None
sep = "#"

prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."
while True:
    width, height = 512, 512
    try:
        prompt = input("Enter a prompt(exit to quit): ")
    except:
        break
    if prompt == "exit":
        break
    elif prompt.startswith("save") and image is not None:
        img = prompt[4:].strip()
        if len(img) <= 0:
            img = "output"
        if not img.endswith(".png"):
            img += ".png"
        image.save(img)
        print("Image saved to :" + img)
        continue
    elif prompt.find(sep) > 0:
        # 3#1280,640#...
        try:
            parts = prompt.split(sep)
            steps, prompt = int(parts[0]), parts[-1]
            if len(parts) >= 3:
                size = parts[1].split(",")
                width, height = int(size[0]), int(size[1])
        except Exception as e:
            print(e)
            continue
    if len(prompt) <= 0:
        continue
    session = pipe(prompt=prompt, num_inference_steps=steps, guidance_scale=0.0, height=height, width=width)
    image = session.images[0]
    image.show()
