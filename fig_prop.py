import streamlit as st


color_palette = [
    'rgb(31, 119, 180)',   # Blue
    'rgb(253, 125, 12)',   # Orange
    'rgb(44, 160, 44)',    # Green
    'rgb(214, 39, 40)',    # Red
    'rgb(148, 103, 189)',  # Purple
    'rgb(140, 86, 75)',     # Brown
    'rgb(255, 187, 120)', # Light Orange
    'rgb(152, 223, 138)', # Light Green
    'rgb(255, 152, 150)', # Light Red
    'rgb(197, 176, 213)', # Light Purple
    'rgb(179, 222, 105)', # Lime Green
    'rgb(102, 0, 153)', # Dark Purple
    'rgb(169, 212, 204)', # Light Blue
    'rgb(242, 195, 219)', # Light Pink
    'rgb(245, 223, 143)', # Light Yellow
    'rgb(160, 160, 160)', # Gray
    'rgb(255, 0, 128)' # Intense Magenta
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
        #title=fig_props.get('title'),
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
            exponentformat = 'none',
            showline=True,
            linewidth=3,
            linecolor='black'
        ),
        yaxis = dict(
            showexponent = 'all',
            exponentformat = 'none',
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
            size=16
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