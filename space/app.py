import math
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr


class CoordNet(torch.nn.Module):
    """Periodic coordinate MLP. modes=1 -> plain base encoding; modes=K -> Fourier features."""
    def __init__(self, modes=1, width=64, depth=4):
        super().__init__()
        self.modes = modes
        layers, d = [], 2 * modes
        for _ in range(depth):
            layers += [torch.nn.Linear(d, width), torch.nn.Tanh()]; d = width
        layers += [torch.nn.Linear(width, 1)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        ks = torch.arange(1, self.modes + 1, dtype=x.dtype)
        ang = 2 * math.pi * x[:, None] * ks[None, :]
        return self.net(torch.cat([torch.sin(ang), torch.cos(ang)], 1)).squeeze(-1)


def run(m, K, steps):
    m, K, steps = int(m), int(K), int(steps)
    A = 0.3

    def us(x):
        return torch.sin(2 * math.pi * x) + A * torch.sin(2 * math.pi * m * x)

    def g(x):
        return 2 * math.pi * torch.cos(2 * math.pi * x) + A * 2 * math.pi * m * torch.cos(2 * math.pi * m * x)

    xg = torch.linspace(0, 1, 401)[:-1]
    tgt = us(xg)
    x0 = torch.zeros(1)

    def train(net):
        opt = torch.optim.Adam(net.parameters(), lr=3e-3)
        for _ in range(steps):
            xf = torch.rand(512)
            x = xf.clone().requires_grad_(True)
            u = net(x)
            ux = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
            loss = ((ux - g(xf)) ** 2).mean() + 10 * ((net(x0) - us(x0)) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            return net(xg).numpy()

    torch.manual_seed(0); p = train(CoordNet(1))
    torch.manual_seed(0); f = train(CoordNet(K))
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    ax.plot(xg.numpy(), tgt.numpy(), color="0.55", lw=3, label="target")
    ax.plot(xg.numpy(), p, color="#D55E00", lw=1.8, label="plain PINN")
    ax.plot(xg.numpy(), f, color="#0072B2", lw=1.8, label="Fourier PINN")
    ax.set_xlabel("x"); ax.set_ylabel("u(x)")
    ax.set_title(f"high mode m={m},  Fourier modes K={K},  steps={steps}")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


with gr.Blocks(title="Fourier vs plain PINN") as demo:
    gr.Markdown(
        "# Fourier vs plain PINN: catching the fine detail\n"
        "Both networks are trained only on the PDE residual of `u'(x)=g(x)` for a "
        "smooth wave carrying a fine high-frequency ripple. The plain network is "
        "spectrally biased and stays smooth; the Fourier-feature network captures "
        "the ripple. Move the sliders and press **Run**.")
    with gr.Row():
        m = gr.Slider(4, 20, value=14, step=1, label="high mode m (fine-detail frequency)")
        K = gr.Slider(2, 24, value=16, step=1, label="Fourier feature modes K")
        steps = gr.Slider(500, 4000, value=1500, step=100, label="training steps")
    btn = gr.Button("Run", variant="primary")
    out = gr.Plot()
    btn.click(run, [m, K, steps], out)

if __name__ == "__main__":
    demo.launch()
