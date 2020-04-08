#!/usr/bin/env python
# coding: utf-8

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

# HTML page Layout
app.layout = html.Div(className="container", children=[
    html.Div(className="row", children=[
        html.Div(className="eight columns", children=[
            html.H2('Chromatographie',
                    style={"color": "#2980b9", "borderBottom": "solid 2px #2980b9"}),
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
        html.Div(className="four columns", children=[
            html.A(
                html.Img(
                    src="http://gvallver.perso.univ-pau.fr/img/logo_uppa.png",
                    height="100px",
                ),
                href="https://www.univ-pau.fr"
            )
        ]),
    ]),
    html.Div(
        dcc.Graph(id='graph'),
        # style={"borderTop": "solid 1px #2980b9", "marginTop": "20px"}
    ),
    html.Div([
        html.Div(className="row", children=[
            html.Div(className="six columns", children=[
                html.P(children=[
                    "Hosted on ",
                    html.A("heroku", href="https://www.heroku.com/")
                ])
            ]),
            html.Div(className="six columns", children=[
                html.P(children=[
                    html.A("Germain Salvato Vallverdu",
                           href="https://gsalvatovallverdu.gitlab.io")
                ])
            ], style={"textAlign": "right"})
        ]),
    ], style={"borderTop": "solid 1px #2980b9", "paddingTop": "20px",
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
    pos = [np.random.uniform(1, 1.1),
           np.random.uniform(3, 4),
           np.random.uniform(4, 5),
           np.random.uniform(10, 12)]
    npos = len(pos)
    amp = np.random.uniform(1, 3, size=npos)
    width = np.random.uniform(0.05, 0.15, size=npos)

    pics = [(a, p, w) for a, p, w in zip(amp, pos, width)]

    tps = np.linspace(0, 15, 1000)
    spectre = make_chromato(tps, pics)

    fig = px.line(
        x=tps, y=spectre,
        range_x=(0, 15),
        range_y=(spectre.min(), 1.2 * spectre.max()),
        title="Chromatogramme %s" % value,
        labels={"x": "temps (min)", "y": "Intensité"},
        template="plotly_white",
        color_discrete_sequence=["#2980b9"],
    )

    fig.update_layout(
        # width=943,
        height=666,
        yaxis={'scaleanchor': 'x'},
        font=dict(
            # family="Arial",
            size=20,
            color="#2c3e50"
        )
    )

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
