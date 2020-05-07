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
data_file = "assets/data/config.yml"
with open(data_file, "r") as fdata:
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
    html.Div(id="integrale"),
    html.Div(
        dcc.Graph(id='vd-graph'),
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

def van_deemter(x, a=1, b=1, c=1):
    """ Van deemter function """
    return a + b / x + c * x

def make_chromato(t, peaks, noise=0.05):
    """ produce a chromatogram 

    Args:
        t (float): a numpy array of float corresponding to the time
        peaks (list of tuples): list of peaks each peak is defined from a tuple 
            such as (amplitude, position, width)
        noise (float): amplitude of a gaussian noise added to the spectra

    Returns
        the spectra and the integral of each peaks.
    """
    chromato = np.zeros(t.shape)
    integrals = list()
    for amp, pos, width in peaks:
        y = skewed(t, pos, width, alpha=2, a=amp)
        integrals.append(np.trapz(y, t))
        chromato += y
        # chromato += amp * normpdf(t, mu, sigma)

    # add some noise
    chromato += np.random.normal(loc=0, scale=noise, size=t.shape)

    return chromato, integrals


@app.callback(
    [Output('graph', 'figure'),
     Output('vd-graph', 'figure'),
     Output('integrale', 'children')],
    [Input('submit', 'n_clicks')],
    [State('student-id', 'value')]
)
def display_graph(n_clicks, value):
    """ Display the random chromatogram """

    # abscissa : time
    chromato_data = data["chromatogram"]
    tmin = chromato_data["tmin"]
    tmax = chromato_data["tmax"]
    npts = chromato_data["npts"]
    tps = np.linspace(tmin, tmax, npts)

    if value is None:
        fig = {"layout": dict(height=666, xaxis={"range": (tmin, tmax)},
                              yaxis={"range": (0, 4)})}
        return fig, {}, []

    # set up a seed
    np.random.seed(int(value))

    # random peaks from 'spectre.yml' data
    peaks = list()
    for peak in data["peaks"]:
        pos = np.random.uniform(peak["pos"]["min"], peak["pos"]["max"])
        amp = np.random.uniform(peak["amp"]["min"], peak["amp"]["max"])
        width = np.random.uniform(peak["width"]["min"], peak["width"]["max"])
        peaks.append((amp, pos, width))

    # build spectre and print integrals
    spectre, integrals = make_chromato(tps, peaks)
    items = list()
    for i, integral in enumerate(integrals):
        li = html.Li(children=[
            html.B("peak %d: " % (i + 1)),
            "%f" % integral,
        ])
        items.append(li)
    div_integrals = [html.H5("Integrales des pics"), html.Ul(items)]

    # plot of the chromatogram
    fig = px.line(
        x=tps, y=spectre,
        title="Chromatogramme %s" % value,
        labels={"x": chromato_data["xlabel"], "y": chromato_data["ylabel"]},
        template="plotly_white",
        color_discrete_sequence=["#2980b9"],
    )

    fig.update_layout(
        # width=943,
        height=666,
        yaxis={"range": (spectre.min(), 1.2 * spectre.max())},
        xaxis={"range": (tmin, tmax)},
        font=dict(
            # family="Arial",
            size=20,
            color="#2c3e50"
        )
    )

    # Van Deemter plot
    vd_data = data["van_demeter"]
    x = np.linspace(vd_data["xmin"], vd_data["xmax"], 1000)
    a = np.random.uniform(vd_data["a"]["min"], vd_data["a"]["min"])
    b = np.random.uniform(vd_data["b"]["min"], vd_data["b"]["min"])
    c = np.random.uniform(vd_data["c"]["min"], vd_data["c"]["min"])
    vd_fig = px.line(
        x=x, y=van_deemter(x, a, b, c),
        title="Van Deemter",
        labels={"x": vd_data["xlabel"], "y": vd_data["xlabel"]},
        template="plotly_white",
        #color_discrete_sequence=["#2980b9"],
    )
    vd_fig.update_layout(
        font=dict(size=20, color="#2c3e50")
    )

    return fig, vd_fig, div_integrals


if __name__ == '__main__':
    app.run_server(debug=True)
