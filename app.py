import tkinter as tk
import customtkinter as ctk

from PIL import ImageTk

from auth_token import AuthToken
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

#Create the app
app = tk.Tk()
app.geometry("532x622")
app.title("Stable Bud")
ctk.set_appearance_mode("dark")

prompt = ctk.CTkEntry(master=app, height=40, width=512, text_color="black", fg_color="white", font=("Arial", 20))
prompt.place(x=10, y=10)

lmain = ctk.CTkLabel(master=app, height=512, width=512)
lmain.place(x=10, y=110)


modelid = "ComVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16", torch_dtypes=torch.float16)
device='cuda'
pipe.to(device)
def generate():
    pass

trigger = ctk.CTkButton(master=app,height=40, width=120, text_color="white", fg_color="blue", font=("Arial", 20), command=generate)
trigger.configure(text="Generate")
trigger.place(x=206, y=60)


app.mainloop()