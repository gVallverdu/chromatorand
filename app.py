#!/usr/bin/env python
# coding: utf-8

import yaml
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px

import numpy as np
from scipy.special import erf

external_stylesheets = [
    'https://codepen.io/chriddyp/pen/bWLwgP.css',
    "https://use.fontawesome.com/releases/v5.8.1/css/all.css",
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# load yaml file to define the spectra
spectre_data = "assets/data/spectre.yml"
with open(spectre_data, "r") as fdata:
    data = yaml.load(fdata, Loader=yaml.SafeLoader)

# HTML page Layout
app.layout = html.Div(className="container", children=[
    html.Div([
        html.A(
            id="github-link",
            href="https://github.com/gVallverdu/chromatorand",
            children=[
                html.Span(id="github-icon", className="fab fa-github fa-2x",
                          style={"verticalAlign": "bottom"}),
                " View on GitHub",
            ],
            style={"color": "#7f8c8d", "textDecoration": "none",
                   "display": "block", "width": "160px",
                   "border": "solid 1px #7f8c8d", "borderRadius": "4px",
                   "padding": "5px", "textAlign": "center", },
        ),
    ], style={"float": "right"}
    ),
    html.Div(className="row", children=[
        html.Div(children=[
            html.H2('Chromatographie',
                    style={"color": "#2980b9", "borderBottom": "solid 2px #2980b9",
                           "paddingTop": "30px"}),
            html.H5("Controle du XX avril"),
            html.Label("Saisir votre numéro étudiant :"),
            dcc.Input(id="student-id", type="number", debounce=True,
                      placeholder=0),
            html.Button('Submit', id='submit', n_clicks=0,
                        className="button-primary",
                        style={"marginLeft": "20px"}),
            html.P("Cliquer sur l'appareil photo pour télécharger l'image.",
                   style={"marginTop": "20px"})
        ]),

    ]),
    html.Div(
        dcc.Graph(id='graph'),
    ),
    html.Div([
        html.Div(className="row", children=[
            html.Div(className="six columns", children=[
                html.A(
                    html.Img(
                        src="http://gvallver.perso.univ-pau.fr/img/logo_uppa.png",
                        height="50px",
                    ),
                    href="https://www.univ-pau.fr"
                )
            ]),
            html.Div(className="six columns", children=[
                html.P(children=[
                    html.A("Germain Salvato Vallverdu",
                           href="https://gsalvatovallverdu.gitlab.io",
                           style={"color": "#7f8c8d"})
                ]),
            ], style={"textAlign": "right", "paddingTop": "10px"})
        ]),
    ], style={"borderTop": "solid 2px #7f8c8d",
              "marginTop": "10px", "fontSize": "small"})
])

# ------------------------------------------------------------------------------
# utility functions


def normpdf(x, mu, sigma):
    """ gaussian function located at mu and half width sigma """
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2 * sigma**2))


def normcdf(x, mu, sigma, alpha=1):
    """ gaussian cdf at located at mu of width sigma and height alpha """
    return 0.5 * (1 + erf(alpha * (x - mu) / (sigma * np.sqrt(2))))


def skewed(x, mu, sigma, alpha, a):
    """ skewed distribution located at mu of width sigma amplitude a and 
    skewness alpha """
    return a * normpdf(x, mu, sigma) * normcdf(x, mu, sigma, alpha)


def make_chromato(t, pics, noise=0.05):
    """ produce a chromatogram 

    Args:
        t (float): a numpy array of float corresponding to the time
        pics (list of tuples): list of pics each pic is defined from a tuple 
            such as (amplitude, position, width)
        noise (float): amplitude of a gaussian noise added to the spectra
    """
    chromato = np.zeros(t.shape)
    for amp, mu, sigma in pics:
        chromato += skewed(t, mu, sigma, alpha=2, a=amp)
        # chromato += amp * normpdf(t, mu, sigma)

    # add some noise
    chromato += np.random.normal(loc=0, scale=noise, size=t.shape)

    return chromato


@app.callback(
    Output('graph', 'figure'),
    [Input('submit', 'n_clicks')],
    [State('student-id', 'value')]
)
def display_graph(n_clicks, value):
    """ Display the random chromatogram """

    # abscissa : time
    tmax = 15
    tps = np.linspace(0, tmax, 1000)

    if value is None:
        fig = {"layout": dict(height=666, xaxis={"range": (0, tmax)},
                              yaxis={"range": (0, 4)})}
        return fig
    # set up a seed
    np.random.seed(int(value))

    # random pics from 'spectre.yml' data
    pics = list()
    for pic in data:
        pos = np.random.uniform(pic["pos"]["min"], pic["pos"]["max"])
        amp = np.random.uniform(pic["amp"]["min"], pic["amp"]["max"])
        width = np.random.uniform(pic["width"]["min"], pic["width"]["max"])
        pics.append((amp, pos, width))

    # build spectre
    spectre = make_chromato(tps, pics)

    # plot
    fig = px.line(
        x=tps, y=spectre,
        title="Chromatogramme %s" % value,
        labels={"x": "temps (min)", "y": "Intensité"},
        template="plotly_white",
        color_discrete_sequence=["#2980b9"],
    )

    fig.update_layout(
        # width=943,
        height=666,
        yaxis={"range": (spectre.min(), 1.2 * spectre.max())},
        xaxis={"range": (0, tmax)},
        font=dict(
            # family="Arial",
            size=20,
            color="#2c3e50"
        )
    )

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
