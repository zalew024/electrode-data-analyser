import streamlit as st


color_palette = [
    'rgb(0, 0, 0)',   # Black
    'rgb(255, 0, 0)',   # Red
    'rgb(0, 0, 255)',    # Blue
    'rgb(255, 0, 255)',    # Magenta
    'rgb(0, 128, 0)',  # Green
    'rgb(30, 0, 128)',     # Blue-Purple
    'rgb(255, 128, 0)', # Orange
    'rgb(128, 0, 128)', # Purple
    'rgb(0, 128, 128)', # Teal
    'rgb(128, 128, 0)', # Olive
    'rgb(0, 255, 0)', # Lime Green
    'rgb(255, 0, 128)', # Pink
    'rgb(128, 0, 0)', # Maroon
    'rgb(0, 255, 255)', # Cyan
    'rgb(255, 255, 0)', # Yellow
    'rgb(128, 128, 128)', # Gray
    'rgb(128, 0, 255)', # Light Purple 
    'rgb(43, 99, 162)', # Light Blue 
    'rgb(255, 178, 255)' # Light Magenta 
]
marker_symbols = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up']


def data_to_plot(data, key):
    return st.multiselect(
        'Select data to plot:',
        list(data),
        list(data),
        key=key)

def download_fig(img):
    return st.download_button(
        label="Export figure for publication",
        data=img,
        file_name="figure.png",
        mime="image/png"
    )


@st.cache_data
def export_fig(fig, **fig_props):
    fig.update_layout(
        showlegend=fig_props.get('legend'),
        margin=dict(
            l=230,
            r=80,
            b=140,
            t=80
        ),
        font=dict(
            size=25
        ),
        xaxis = dict(
            showexponent = 'all',
            exponentformat = 'B',
            showline=True,
            linewidth=3,
            linecolor='black'
        ),
        yaxis = dict(
            range=fig_props.get('range'),
            showexponent = 'all',
            exponentformat = 'B',
            showline=True,
            linewidth=3,
            linecolor='black',
            title_standoff=25
        ),
        colorway=color_palette
    )
    fig.update_shapes(
        selector=dict(
            name='vertical_line'
        ), 
        line_color='#000000', 
        line_width=1
    )
    fig.update_xaxes(
        title_text=fig_props.get('x_label'),
        minor=dict(
            ticklen=5,
            showgrid=True,
            gridcolor='#b5b3b3'
        ),
        showgrid=True,
        griddash='dot',
        gridcolor='#b5b3b3'
    )
    fig.update_yaxes(
        title_text=fig_props.get('y_label'),
        minor=dict(
            ticklen=5,
            showgrid=True,
            gridcolor='#b5b3b3'
        ),
        showgrid=True,
        griddash='dot',
        gridcolor='#b5b3b3'
    )
    fig.update_traces(
        line=dict(
            width=fig_props.get('width')
        ), 
        marker=dict(
            size=fig_props.get('marker')
        )
    )
    for i, trace in enumerate(fig.data):
        trace.marker.symbol = marker_symbols[i % len(marker_symbols)]
    fig_x = fig.to_image(width=1400, height=1000, format='png', engine="kaleido")
    return fig_x

@st.cache_data
def show_fig(fig, **fig_props):
    fig.update_layout(
        title={
            'text': fig_props.get('title'),
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        showlegend=fig_props.get('legend'),
        margin=dict(
            l=50,
            r=50,
            t=100,
            b=50 
        ),
        font=dict(
            size=24
        ),
        xaxis = dict(
            showexponent = 'all',
            exponentformat = 'e'
        ),
        yaxis = dict(
            showexponent = 'all',
            exponentformat = 'e'
        )
    )
    fig.update_xaxes(
        title_text=fig_props.get('x_label'),
        minor=dict(
            ticklen=5,
            showgrid=True
        ),
        showgrid=True,
        griddash='dot'
    )
    fig.update_yaxes(
        title_text=fig_props.get('y_label'),
        minor=dict(
            ticklen=5,
            showgrid=True
        ),
        showgrid=True,
        griddash='dot'
    )
    st.plotly_chart(fig)