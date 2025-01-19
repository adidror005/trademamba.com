import dash
from dash import html
import dash_bootstrap_components as dbc

# Sample Data for testing
data = [
    {"headline": "Stock prices surge", "pred": "Buy", "created_at": "2025-01-19"},
    {"headline": "Market crashes", "pred": "Sell", "created_at": "2025-01-18"},
    {"headline": "Steady growth", "pred": "Hold", "created_at": "2025-01-17"},
]

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout
app.layout = html.Div(
    style={"margin": "20px"},
    children=[
        html.H1(
            "Prediction Table Example",
            style={"textAlign": "center", "marginBottom": "20px", "fontWeight": "bold"},
        ),
        dbc.Table(
            [
                html.Thead(
                    html.Tr([html.Th("Headline"), html.Th("Prediction"), html.Th("Created At")])
                ),
                html.Tbody(
                    [
                        html.Tr(
                            [
                                html.Td(
                                    row["headline"],
                                    style={"whiteSpace": "normal", "wordBreak": "break-word"}
                                ),
                                html.Td(
                                    row["pred"],
                                    style={"textAlign": "center",
                                            "backgroundColor": "lightblue" if row["pred"] == "Buy"
                                            else "lightcoral" if row["pred"] == "Sell"
                                            else "lightyellow"}
                                ),
                                html.Td(
                                    row["created_at"],
                                    style={"fontSize": "12px"}
                                ),
                            ]
                        )
                        for row in data  # Loop through the data list to create rows
                    ]
                ),
            ],
            striped=True,
            bordered=True,
            hover=True,
            responsive=True,
        ),
    ],
)

if __name__ == "__main__":
    app.run_server(debug=True)
