import dash
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
import dash_player
from dash import html, dcc, Input, Output, State, _dash_renderer
import plotly.express as px
_dash_renderer._set_react_version("18.2.0")
from pages.mnist_page import layout as mnist_layout,img
from pages.sentiment import  layout as sentiment_layout, df_text,generate_table,ROWS_PER_PAGE

# Data/Images/etc...
##############################################################################
trade_mamba_logo = 'https://raw.githubusercontent.com/adidror005/youtube-videos/refs/heads/main/trade_mamba_new.jpg'
llm_tutorials_img = 'https://raw.githubusercontent.com/adidror005/youtube-videos/refs/heads/main/Add%20a%20heading%20(11).png'
ibkr_tutorials_img = 'https://raw.githubusercontent.com/adidror005/youtube-videos/refs/heads/main/1.png'

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
    "zIndex": 1000,
}

SIDEBAR_HIDDEN_STYLE = {
    "display": "none"
}

CONTENT_STYLE = {
    "margin-left": "18rem",
    "padding": "2rem 1rem",
    "transition": "margin-left 0.3s ease-in-out",
}

CONTENT_FULL_STYLE = {
    "margin-left": "0",
    "padding": "2rem 1rem",
    "transition": "margin-left 0.3s ease-in-out",
}

# Components
##############################################################################
ibkr_playlist_link = html.A(
    "Click Here For Full Playlist!",
    href='https://www.youtube.com/playlist?list=PLCZZtBmmgxn8CFKysCkcl-B1tqRgCCNIX',
    target="_blank"
)

ibkr_youtube_video = dash_player.DashPlayer(
    id="ibkr-player",
    url="https://www.youtube.com/watch?v=_AASJZyNcXQ",
    controls=True,
    width="100%",
    height="auto",
    style={"aspectRatio": "16/9", "paddingLeft": "10px"}
)

llm_youtube_video = dash_player.DashPlayer(
    id="llm-player",
    url="https://www.youtube.com/watch?v=YJNbgusTSF0",
    controls=True,
    width="100%",
    height="auto",
    style={"aspectRatio": "16/9", "paddingLeft": "10px"}
)

ibkr_accordion = dmc.Accordion(
    children=[
        dmc.AccordionItem(
            [
                dmc.AccordionControl(
                    "Click Here for Videos",
                    style={
                        "backgroundColor": "#007bff",
                        "color": "white",
                        "borderRadius": "5px",
                        "padding": "10px 20px",
                        "cursor": "pointer",
                        "fontWeight": "bold",
                        "textAlign": "center",
                    }
                ),
                dmc.AccordionPanel([
                    "Popular Video", ibkr_youtube_video, ibkr_playlist_link
                ]),
            ],
            value="info",
        )
    ],
    disableChevronRotation=True,
    variant="separated",
)

llm_accordion = dmc.Accordion(
    children=[
        dmc.AccordionItem(
            [
                dmc.AccordionControl(
                    "Click Here for Videos",
                    style={
                        "backgroundColor": "#007bff",
                        "color": "white",
                        "borderRadius": "5px",
                        "padding": "10px 20px",
                        "cursor": "pointer",
                        "fontWeight": "bold",
                        "textAlign": "center",
                    }
                ),
                dmc.AccordionPanel([
                    "Popular Video", llm_youtube_video
                ]),
            ],
            value="info",
        )
    ],
    disableChevronRotation=True,
    variant="separated",
)

ibkr_tutorial_card = dmc.Card(
    children=[
        html.H1("Interactive Brokers in Python"),
        dmc.CardSection(
            dmc.Image(
                src=ibkr_tutorials_img,
                #h=360, w=640,
                alt="Interactive Brokers Python Tutorials",
            )
        ),
        dmc.Group(
            [
                dmc.Text("Interactive Brokers Python Tutorials", fw=500),
                dmc.Badge("New Video", color="pink"),
            ],
            justify="space-between",
            mt="md",
            mb="xs",
        ),
        dmc.Text(
            "100+ Videos on Algorithmic Trading with Interactive Brokers in Python",
            size="lg",
            c="dimmed",
        ),
        ibkr_accordion
    ],
    withBorder=True,
    shadow="sm",
    radius="md",
   # w=960,
)

llm_tutorial_card = dmc.Card(
    children=[
        html.H1("LLM Applications in Finance"),
        dmc.CardSection(
            dmc.Image(
                src=llm_tutorials_img,
                #h=360, w=640,
                alt="LLM Applications in Finance",
            )
        ),
        dmc.Group(
            [
                dmc.Text("LLM Tutorials", fw=500),
                dmc.Badge("New Video", color="pink"),
            ],
            justify="space-between",
            mt="md",
            mb="xs",
        ),
        dmc.Text(
            "Videos on LLMs from fine-tuning, to API usage, to LangChain, ChatGPT, etc.",
            size="lg",
            c="dimmed",
        ),
        llm_accordion
    ],
    withBorder=True,
    shadow="sm",
    radius="md",
    #w=960,
)

# Layout
##############################################################################
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dmc.styles.ALL])
app.config.suppress_callback_exceptions = True

sidebar = html.Div(
    [
        html.H2("Navigation", className="display-4"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", active="exact"),
                dbc.NavLink("Links", href="/page-1", active="exact"),
                dbc.NavLink("Apps", href="/page-2", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    id="sidebar",
    style=SIDEBAR_STYLE,
)

content = html.Div(
    id="content",
    style=CONTENT_STYLE,
)

app.layout = dmc.MantineProvider(
    html.Div([
        dcc.Store(id="sidebar-toggle", data=True),
        dcc.Location(id="url"),
        sidebar,
        content
    ])
)

# Callbacks
##############################################################################
@app.callback(
    [Output("sidebar", "style"), Output("content", "style"), Output("sidebar-toggle", "data")],
    [Input("toggle-button", "n_clicks")],
    [State("sidebar-toggle", "data")],
)
def toggle_sidebar(n_clicks, is_sidebar_visible):
    if n_clicks:
        new_visibility = not is_sidebar_visible
        if new_visibility:
            return SIDEBAR_STYLE, CONTENT_STYLE, new_visibility
        else:
            return SIDEBAR_HIDDEN_STYLE, CONTENT_FULL_STYLE, new_visibility
    return SIDEBAR_STYLE, CONTENT_STYLE, is_sidebar_visible

@app.callback(
    [Output("sidebar", "style",allow_duplicate=True), Output("content", "style",allow_duplicate=True), Output("sidebar-toggle", "data",allow_duplicate=True)],
    Input("url", "pathname"),
    prevent_initial_call=True,
)
def handle_navigation(pathname):
    if pathname == "/mnist-app":
        return SIDEBAR_HIDDEN_STYLE, CONTENT_FULL_STYLE, False
    elif pathname == "/sentiment-app":
        return SIDEBAR_HIDDEN_STYLE, CONTENT_FULL_STYLE, False
    else:
        return SIDEBAR_STYLE, CONTENT_STYLE, True

@app.callback(
    Output("content", "children"),
    Input("url", "pathname"),
)
def render_page_content(pathname):
    if pathname == "/page-1":
        return html.Div([
            html.H1("Links Page"),
            html.A("YouTube Channel", href='https://www.youtube.com/channel/UCZHN0IOGmmvY6JtquMoEn9w', target="_blank"),
            html.Br(),
            html.A("Medium Blog", href='https://medium.com/@trademamba', target="_blank"),
            html.Br(),
            html.A("X",href="https://x.com/AdiDror6", target="_blank")
        ])
    elif pathname == "/page-2":
        return html.Div([
            html.H1("Apps Page"),
            dbc.Button("Go to MNIST App", id="mnist-btn", n_clicks=0, href="/mnist-app", color="primary", className="mb-3"),
            html.Br(),
            dbc.Button("Go to Stock News Sentiment App", id="sentiment-btn", n_clicks=0, href="/sentiment-app", color="primary",
                       className="mb-3"),

        ])
    elif pathname == "/mnist-app":
        return html.Div([
            dbc.Row(dbc.Col(mnist_layout)),
            dbc.Button("Back to Main Menu", href="/", color="secondary", className="mt-3"),
        ])
    elif pathname == "/sentiment-app":
        return html.Div([
            dbc.Row(dbc.Col(sentiment_layout)),
            dbc.Button("Back to Main Menu", href="/", color="secondary", className="mt-3"),
        ])
    elif pathname == "/":
        return html.Div([
            html.H1("Trade Mamba!"),
            dbc.Row(
                dbc.Col(
                    html.Div(
                        """
                        Welcome to Trade Mamba's website where I have tutorials and data science apps.
                        """,
                        className="lead",
                    )
                ),
                className="mb-4",
            ),
            dbc.Button("Click Here to See More!", id="toggle-button", n_clicks=1, className="mb-3"),
            dbc.Row(
                dbc.Col(
                    html.Div(
                        """
                        My YouTube channel explores the intersection of data science, machine learning, and algorithmic trading. 
                        Gain insights into financial data analysis, predictive modeling, and advanced trading strategies.
                        """,
                        className="lead",
                    )
                ),
                className="mb-4",
            ),
            dbc.Row(dbc.Col(html.H1("Python Tutorial Videos:"))),
            dbc.Row(
                children=[
                    dbc.Col(ibkr_tutorial_card, xs=11, sm=9, md=5, lg=5, xl=6),
                    dbc.Col(llm_tutorial_card, xs=11, sm=9, md=5, lg=5, xl=6),
                ]
            ),
            dbc.Row(dbc.Col(html.Div([html.Img(src=trade_mamba_logo, width=100)]), xs=12, sm=12, md=12, lg=12, xl=12)),
        ])
    return html.Div([
        html.H1("404: Page not found"),
        html.P("The page you are looking for does not exist."),
    ])

@app.callback(
    Output("graph-picture", "figure"),
    Input("reset-button", "n_clicks"),
    prevent_initial_call=True,
)
def reset_graph(n_clicks):
    fig = px.imshow(img, height=420)
    fig = px.imshow(img,height=420)
    fig.update_layout(
        dragmode="drawopenpath",
        newshape=dict(line_color='red', line_width=45)  # Set color to red and width to 5
    )
    fig.update_layout(
        dragmode="drawopenpath",
        newshape=dict(line_color='red', line_width=45),
        margin=dict(l=0, r=0, t=0, b=0),  # Remove all margins
        xaxis=dict(visible=False, fixedrange=True),  # Hide x-axis
        yaxis=dict(visible=False, fixedrange=True)  # Hide y-axis
    )
    return fig

# Callbacks for table and pagination
@app.callback(
    Output("table-container", "children"),
    Output("current-page", "children"),
    [Input("stock-dropdown", "value"), Input("prev-page", "n_clicks"), Input("next-page", "n_clicks")],
    [State("current-page", "children")],
)
def update_table(selected_symbol, prev_clicks, next_clicks, current_page):
    if not current_page:
        current_page = "Page 1"
    page_number = int(current_page.split()[-1]) - 1

    # Adjust page based on navigation buttons
    page_number = max(0, page_number + (1 if next_clicks > prev_clicks else -1))

    if selected_symbol:
        filtered_df = df_text[df_text["symbol"] == selected_symbol]
    else:
        filtered_df = df_text

    # Calculate max page
    max_page = (len(filtered_df) - 1) // ROWS_PER_PAGE
    page_number = min(page_number, max_page)

    return generate_table(filtered_df, page=page_number), f"Page {page_number + 1}"
