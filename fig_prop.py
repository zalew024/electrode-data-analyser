import streamlit as st


color_palette = [
    'rgb(31, 119, 180)',   # Blue
    'rgb(255, 127, 14)',   # Orange
    'rgb(44, 160, 44)',    # Green
    'rgb(214, 39, 40)',    # Red
    'rgb(148, 103, 189)',  # Purple
    'rgb(140, 86, 75)'     # Brown
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
        title=fig_props.get('title'),
        showlegend=fig_props.get('legend'),
        margin=dict(
            l=150,
            r=150,
            b=130,
            t=130
        ),
        font=dict(
            size=22
        ),
        xaxis = dict(
            showexponent = 'all',
            exponentformat = 'e'
        ),
        yaxis = dict(
            showexponent = 'all',
            exponentformat = 'e'
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
            size=14
        )
    )
    for i, trace in enumerate(fig.data):
        trace.marker.symbol = marker_symbols[i % len(marker_symbols)]
    fig_x = fig.to_image(scale=10, format='png', engine="kaleido")
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
            ticklen=5
        ),
        showgrid=True,
        griddash='dot'
    )
    st.plotly_chart(fig)