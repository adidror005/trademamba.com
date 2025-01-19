import dash_ag_grid as dag
import dash
from dash import html, dcc
import pandas as pd


df_text = pd.read_csv("data/")
app = dash.Dash(__name__)

app.layout = html.Div([
    dag.AgGrid(
        id='my-grid',
        columnDefs=[
            {'field': 'headline'},
            {
                'field': 'pred',
                'cellStyle': {
                    'styleConditions': [
                        {'condition': "params.value == 'Buy'", 'style': {'backgroundColor': 'green'}},  # Buy
                        {'condition': "params.value == 'Sell'", 'style': {'backgroundColor': 'red'}}, # Sell
                        {'condition': "params.value == 'Hold'", 'style': {'backgroundColor': 'yellow'}}  # Hold
                    ]
                }
            }
        ],
        rowData=df_text.to_dict('records'),
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)