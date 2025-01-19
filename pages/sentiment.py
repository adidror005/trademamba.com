import dash
from dash import html, dcc
import pandas as pd
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

# Load your data
df_text = pd.read_csv("data/df_text_with_predictions.csv")
df_text['symbol']=df_text['symbols'].str.replace("['","").str.replace("']","")


# Rows per page
ROWS_PER_PAGE = 10

def generate_table(dataframe, page=0):
    """Generate an HTML table from a dataframe with conditional styling and pagination."""
    start_row = page * ROWS_PER_PAGE
    end_row = start_row + ROWS_PER_PAGE
    paged_df = dataframe.iloc[start_row:end_row]

    # Conditional styling based on "pred"
    def row_style(pred):
        if pred == "Buy":
            return {"backgroundColor": "green", "color": "white"}
        elif pred == "Sell":
            return {"backgroundColor": "red", "color": "white"}
        elif pred == "Hold":
            return {"backgroundColor": "lightyellow", "color": "black"}
        else:
            return {"backgroundColor": "grey", "color": "white"}

    header = [
        html.Thead(
            html.Tr(
                [
                    html.Th("Headline"),
                    html.Th("Prediction"),
                    html.Th("Created At"),
                ]
            )
        )
    ]

    body = [
        html.Tbody(
            [
                html.Tr(
                    [
                        html.Td(row["headline"], style={"whiteSpace": "normal", "wordBreak": "break-word"}),
                        html.Td(
                            row["pred"],
                            style={
                                "textAlign": "center",
                                **row_style(row["pred"])  # Apply conditional styling here
                            },
                        ),
                        html.Td(row["created_at"], style={"fontSize": "12px"}),
                    ],
                    id={"type": "row", "index": i},
                )
                for i, row in paged_df.iterrows()
            ]
        )
    ]

    return dbc.Table(header + body, striped=True, bordered=True, hover=True, responsive=True)


# App layout
layout = html.Div(
    style={"margin": "20px"},
    children=[
        html.H1(
            "Select a Stock Symbol",
            style={"textAlign": "center", "marginBottom": "20px", "fontWeight": "bold"},
        ),
        dcc.Dropdown(
            id="stock-dropdown",
            options=[
                {"label": symbol, "value": symbol}
                for symbol in ["AMD", "TSLA", "NVDA", "RIVN", "BABA", "MSFT", "META", "UBER"]
            ],
            placeholder="Select a stock symbol",
            style={
                "marginBottom": "20px",
                "width": "100%",
                "maxWidth": "400px",
                "margin": "0 auto",
            },value='TSLA'
        ),
        html.Div(id="table-container"),
        html.Div(
            id="pagination-controls",
            style={"display": "flex", "justifyContent": "center", "marginTop": "20px"},
            children=[
                dbc.Button("Previous", id="prev-page", n_clicks=0, color="primary", className="me-2"),
                dbc.Button("Next", id="next-page", n_clicks=0, color="primary"),
            ],
        ),
        html.Div(id="current-page", style={"textAlign": "center", "marginTop": "10px"}),
    ],
)

