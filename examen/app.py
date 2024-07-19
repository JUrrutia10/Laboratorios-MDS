import gradio as gr
import ex
import pandas as pd


demo = gr.Interface(fn = ex.predapi, # noten como estamos usando la función que generamos anteriormente
                    inputs=gr.File(type="filepath"), 
                    outputs=gr.DataFrame()) # valor de salida

demo.launch(share = True)