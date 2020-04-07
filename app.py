import os

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px

import numpy as np
from scipy.special import erf

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

app.layout = html.Div([
    html.H2('Chromatographie'),
    html.H3("Controle du XX avril"),
    html.P("Saisir votre numéro étudiant :"),
    dcc.Input(id="student-id", type="number", debounce=True,
              placeholder=0),
    html.Button('Submit', id='submit', n_clicks=0),
    html.Div(
        dcc.Graph(id='graph'),
    )
])

def normpdf(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2 * sigma**2))

def normcdf(x, mu, sigma, alpha=1):
    return 0.5 * (1 + erf(alpha * (x - mu) / (sigma * np.sqrt(2))))

def skewed(x, mu, sigma, alpha, a):
    return a * normpdf(x, mu, sigma) * normcdf(x, mu, sigma, alpha)

def make_chromato(t, pics, noise=0.05):

    chromato = np.zeros(t.shape)
    for amp, mu, sigma in pics:
        chromato += skewed(t, mu, sigma, alpha=2, a=amp)
        # chromato += amp * normpdf(t, mu, sigma)

    chromato += np.random.normal(loc=0, scale=noise, size=t.shape)

    return chromato

@app.callback(
    Output('graph', 'figure'),
    [Input('submit', 'n_clicks')],
    [State('student-id', 'value')]
)
def display_graph(n_clicks, value):

    if value is None:
        return {}
    # set up a seed
    np.random.seed(int(value))

    # pics
    pos = [np.random.uniform(3, 4),
           np.random.uniform(4, 5),
           np.random.uniform(6, 7)]
    npos = len(pos)
    amp = np.random.uniform(1, 3, size=npos)
    width = np.random.uniform(0.05, 0.15, size=npos)

    pics = [(a, p, w) for a, p, w in zip(amp, pos, width)]

    tps = np.linspace(0, 15, 1000)
    spectre = make_chromato(tps, pics)

    fig = px.line(
        x=tps, y=spectre, range_x=(0, 15),
        title="Chromatogramme %s" % value,
        labels={"x": "temps (min)", "y": "Intensité"},
        template="plotly_white"
    )

    fig.update_layout(
        font=dict(
            #family="Arial",
            size=20,
            color="#7f7f7f"
        )
    )

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
