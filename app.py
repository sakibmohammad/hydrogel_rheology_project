import gradio as gr
import torch
import numpy as np
import joblib
from model import load_mlp, load_CVAE

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
mlp = load_mlp("Weights/model_regression.pth", device)
cvae = load_CVAE("Weights/model_cvae.pth", device)

x_reg_scaler = joblib.load('Scalers/scaler_regression.pkl')
x_vae_scaler = joblib.load('Scalers/scaler_cvae_x.pkl')
y_vae_scaler = joblib.load('Scalers/scaler_cvae_y.pkl')

def pred_mlp(*args):
    x_values = np.array(args).reshape(1, -1)
    x_values_scaled = x_reg_scaler.transform(x_values)
    x_tensor = torch.tensor(x_values_scaled, dtype=torch.float32).to(device)
    mlp.eval()
    with torch.no_grad():
        output = mlp(x_tensor)

    y_pred = output.cpu().numpy()
    return y_pred[0, 0], y_pred[0, 1]



def pred_cvae(storage_modulus, loss_modulus):
    y_values = np.array([[storage_modulus, loss_modulus]])
    y_values_scaled = y_vae_scaler.transform(y_values)
    y_tensor = torch.tensor(y_values_scaled, dtype=torch.float32).to(device)
    cvae.eval()
    with torch.no_grad():
        z = torch.randn(1, 10).to(device)
        output = cvae.decode(z, y_tensor)

    x_pred = x_vae_scaler.inverse_transform(output.cpu().numpy())
    return tuple(x_pred[0])

options_regression = {
    'Acrylamide Conc. %': [10, 12.5, 15],
    'Bis-acrylamide Conc. %': [0.2, 0.3],
    'Photo-initiator Conc. %': [1.5, 2.0],
    'Layer Height (micron)': [30, 60, 90, 120, 150],
    'Bottom Layer exposure time (s)': [50, 75],
    'Exposure time (s)': [6, 12],
    'Frequency (Hz)': [0.01, 0.0158, 0.0251, 0.0398, 0.0630, 0.1, 0.1584, 0.2511,
                        0.3981, 0.6309, 1, 1.5848, 2.5118, 3.9810, 6.3095, 10]
}

reg_int = gr.Interface(
    fn=pred_mlp,
    inputs=[
        gr.Dropdown(choices=options_regression["Acrylamide Conc. %"], label="Acrylamide Conc. %"),
        gr.Dropdown(choices=options_regression["Bis-acrylamide Conc. %"], label="Bis-acrylamide Conc. %"),
        gr.Dropdown(choices=options_regression["Photo-initiator Conc. %"], label="Photo-initiator Conc. %"),
        gr.Dropdown(choices=options_regression["Layer Height (micron)"], label="Layer Height (micron)"),
        gr.Dropdown(choices=options_regression["Bottom Layer exposure time (s)"], label="Bottom Layer exposure time (s)"),
        gr.Dropdown(choices=options_regression["Exposure time (s)"], label="Exposure time (s)"),
        gr.Dropdown(choices=options_regression["Frequency (Hz)"], label="Frequency (Hz)"),
    ],
    outputs=[
        gr.Number(label="Storage modulus (Pa)", precision= 2),
        gr.Number(label="Loss modulus (Pa)", precision = 2)
    ],
    title="Regression Model: Predict Storage and Loss Modulus",
    submit_btn="Predict",
    theme = gr.themes.Base()
)

cvae_int = gr.Interface(
    fn=pred_cvae,
    inputs=[
        gr.Number(label="Storage modulus (Pa)", precision=2),
        gr.Number(label="Loss modulus (Pa)", precision=2)
    ],
    outputs=[
        gr.Number(label="Acrylamide Conc. %", precision= 2),
        gr.Number(label="Bis-acrylamide Conc. %", precision= 2),
        gr.Number(label="Photo-initiator Conc. %", precision= 2),
        gr.Number(label="Layer Height (micron)", precision= 2),
        gr.Number(label="Bottom Layer exposure time (s)", precision= 2),
        gr.Number(label="Exposure time (s)", precision= 2),
        gr.Number(label="Frequency (Hz)", precision= 2),
    ],
    title="Generative Model: Generate Material Composition",
    submit_btn="Generate",
    theme = gr.themes.Base()
)

app = gr.TabbedInterface([reg_int, cvae_int], ["Regression Model", "Generative Model"])
app.launch(share= True)
