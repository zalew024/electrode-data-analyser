import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from PIL import Image
from scipy.interpolate import PchipInterpolator
from scipy import integrate
from functools import reduce
from fig_prop import export_fig, show_fig, data_to_plot, download_fig


st.set_page_config(
    page_title="Electrode Data Analyser",
    page_icon=":chart_with_upwards_trend:"
)

st.write("""
# Electrode Data Analyser
A web app for experimental data analysis from electrical and electrochemical measurements.
""")


def calc_z():
    st.session_state.impedance = float(Z_interp(st.session_state.frequency)) 

def calc_freq():
    st.session_state.frequency = float(F_interp(st.session_state.impedance))

def get_ele_area(area_key):
    st.divider()
    col1, col2 = st.columns([2, 3])
    with col1:        
        ele_area = st.number_input("**Electrode surface area [mm²]**", value=10.0, step=0.000001, key=area_key, format='%.6f')
    st.divider()
    return ele_area


tab_titles = [
    "Impedance",
    "Double-layer capacitance",
    "Charge-storage capacity",
    "Charge injection limit"
]
tabs = st.tabs(tab_titles)


# Z
with tabs[0]:

    st.markdown("**Plot impedance magnitude from EIS (Electrochemical Impedance Spectroscopy)**")
    df_z = {}
    
    with st.expander("Import data file"):
        try:
            uploaded_files_z = st.file_uploader("Choose a CSV or TXT file", type=['csv', 'txt'], 
                                                accept_multiple_files=True, key='eis_upload')
            for uploaded_file in uploaded_files_z:
                df_z.update({uploaded_file.name[:-4] : pd.read_table(uploaded_file, sep='\t', 
                                                    index_col=None, header=0, usecols=range(2))})
                for name, df in df_z.items():
                    df = df[['|Z| (ohms)', 'Frequency (Hz)']]
        except:
            df_z = {}
            st.error('Imported file does not match the calculation method')
    
    with st.expander("Raw data"):
        data_col = st.columns(2, gap="medium")
        c = 0
        if df_z:
            for name, df in df_z.items():
                df.rename(columns={'|Z| (ohms)': '|Z| (Ω)'}, inplace=True)
                data_col[c].markdown(name)
                data_col[c].dataframe(df)
                c = not c

    if df_z:

        fig = go.Figure()

        options = data_to_plot(df_z.keys(), 'opt_z')

        with st.expander("Figure properties"):
            avg = st.radio("**Traces**", ('independent traces', 'average from selected measurements'),
                            horizontal=True)
            scale = st.radio("**Graph scale**", ('log', 'lin'), horizontal=True)
            fig_props = {
                'title': st.text_input("**Title**", value="Impedance magnitude", key='title_z'),
                'x_label': st.text_input("**X-axis label**", value="f (Hz)", key='xlabel_z'),
                'y_label': st.text_input("**Y-axis label**", value=r'|Z| (Ω⋅cm<sup>2</sup>)', key='ylabel_z'),
                'legend': True if len(options)>1 and avg == 'independent traces' else False,
                'width': 7,
                'marker': 16
            }

        ele_area_z = get_ele_area("ele_z")

        df_z_list = []
        for name, df in df_z.items():
            if name in options:
                df['|Z| (Ω)'] = df['|Z| (Ω)']*(ele_area_z*0.01)
                df.rename(columns = {'Frequency (Hz)':'f (Hz)', 
                                     '|Z| (Ω)':'|Z| (Ωcm) - {}'.format(name)}, 
                                     inplace=True)
                if avg == 'independent traces':
                    fig.add_trace(go.Scatter(x=df['f (Hz)'], y=df['|Z| (Ωcm) - {}'.format(name)], 
                                             mode='lines+markers', name=name))
                else:
                    df_z_list.append(df)
        
        if df_z_list and avg == 'average from selected measurements':
                df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['f (Hz)'],
                                        how='outer'), df_z_list)
                with st.expander("Averaged data"):
                    df_merged = df_merged.sort_values(by=['f (Hz)'], 
                                ascending=False).reset_index(drop=True).interpolate(method='pchip', limit_direction='both', axis=0)
                    av_list = df_merged.columns.values.tolist()
                    av_list.pop(0)
                    df_merged['Average'] = df_merged[av_list].mean(axis=1)
                    st.dataframe(df_merged.style.format("{:.6f}"))
                fig.add_trace(go.Scatter(x=df_merged['f (Hz)'], y=df_merged['Average'],
                                        mode='lines+markers', name=name))
        
        fig.update_xaxes(type='log' if scale == 'log' else 'linear')
        fig.update_yaxes(type='log' if scale == 'log' else 'linear')

        show_fig(fig, **fig_props)

        img_z = export_fig(fig, **fig_props)
        btn = download_fig(img_z)

        st.divider()

        if options:

            if avg == 'average from selected measurements':
                df_interp = df_merged[['f (Hz)', 'Average']]
                col_name = 'Average'
            else:
                interp = st.radio("**Data to interpolate**", options, horizontal=True)
                df_interp = df_z.get(interp)
                col_name = '|Z| (Ωcm) - {}'.format(interp)

            df_interp = df_interp.sort_values(by=[col_name], ascending=True).reset_index(drop=True)
            F_interp = PchipInterpolator(df_interp[col_name], df_interp['f (Hz)'])
            df_interp = df_interp.sort_values(by=['f (Hz)'], ascending=True).reset_index(drop=True)
            Z_interp = PchipInterpolator(df_interp['f (Hz)'], df_interp[col_name])
        
            min_in, max_in = df_interp.min(), df_interp.max()

            col1, col2, col3 = st.columns([4,1,4], gap="large")

            with col1:
                col1_1, col1_2 = st.columns([2,1], gap="small")
                with col1_1:
                    freq = st.number_input('Frequency', min_value=min_in[0], max_value=max_in[0], value=1000.0, 
                                    step=0.01, key='frequency', help="Enter frequency for which corresponding impedance will be found", 
                                    on_change=calc_z)
                with col1_2:
                    freq_units = st.selectbox("", ("Hz",), key='u_freq')
                
            with col2:
                image = Image.open('img/link_icon.png')
                st.image(image, use_column_width='auto')

            with col3:
                col2_1, col2_2 = st.columns([2,1], gap="small")
                with col2_1:
                    z = st.number_input('|Z|', min_value=min_in[1], max_value=max_in[1], value=float(Z_interp(1.0)),
                                    step=0.01, key='impedance', help="Enter impedance for which corresponding frequency will be found", 
                                    on_change=calc_freq)
                with col2_2:
                    z_units = st.selectbox("", ("Ω",), key='u_z')

           
# Cdl
with tabs[1]:

    st.markdown("**Calculate double-layer capacitance from cyclic voltammetry (CV)**")
    df_cdl = {}
    
    with st.expander("Import data file"):
        try:
            uploaded_files_cdl = st.file_uploader("Choose a CSV or TXT file", type=['csv', 'txt'], 
                                                accept_multiple_files=True, key='cdl_upload')
            for uploaded_file in uploaded_files_cdl:
                df_cdl.update({uploaded_file.name[:-4] : pd.read_table(uploaded_file, sep='\t', 
                                                                    index_col=None, header=0, usecols=range(2))})
            for name, df in df_cdl.items():
                    df = df[['Potential (V)', 'Current (A)']]
        except:
            df_cdl = {}
            st.error('Imported file does not match the calculation method')

    if df_cdl:

        fig2_1 = go.Figure()

        options = data_to_plot(df_cdl.keys(), 'opt_cdl')
        
        if len(options)==6:
            v_map_cdl = pd.DataFrame(index=df_cdl.keys(), columns=["integral", "Velocity (mV/s)"])

            for name, df in df_cdl.items():
                v_map_cdl["integral"][name] = integrate.trapz(df['Current (A)'], x=df['Potential (V)'])

            v_map_cdl = v_map_cdl.sort_values(by=["integral"], ascending=True)
            v_map_cdl["Velocity (mV/s)"] = [10, 20, 40, 60, 80, 100]

            with st.expander("Assigned CV velocity"):
                st.write(v_map_cdl["Velocity (mV/s)"])

        with st.expander("Figure properties"):
            fig_props = {
                'title': st.text_input("**Title**", value="Cyclic voltammetry", key='title_cdl'),
                'x_label': st.text_input("**X-axis label**", value=r'E vs. Ag/AgCl<sub>3</sub> (V)', key='xlabel_cdl'),
                'y_label': st.text_input("**Y-axis label**", value=r'J (A⋅cm<sup>-2</sup>)', key='ylabel_cdl'),
                'legend': True if len(options)>1 else False,
                'width': 2
            }

        ele_area_cdl = get_ele_area("ele_cdl")

        cv_range = []
        for name, df in df_cdl.items():
            if name in options:
                label = f'{v_map_cdl["Velocity (mV/s)"][name]} mV/s' if len(options)==6 else name
                cv_range.extend([df['Potential (V)'].min(), df['Potential (V)'].max()])
                fig2_1.add_trace(go.Scatter(x=df['Potential (V)'], y=df['Current (A)']/(ele_area_cdl*0.01), 
                                            mode='lines', name=label, line=dict(width=1)))
            
        if cv_range: 

            col1, col2, col3 = st.columns([2,10,2], gap="small")

            with col2:
                volt = st.slider('Voltage for anodic and cathodic current calculations (V)', float(round(min(cv_range), 3)),
                                float(round(max(cv_range), 3)), value=float(round(sum(cv_range)/len(cv_range), 3)), step=0.001, key='voltage_slider', format='%.3f', 
                                help="Enter voltage for which anodic and cathodic current will be found \n (1 mV resolution)")
            
            #reset
            fig2_1.add_shape(
                type='line',
                xref='x',
                yref='paper',
                x0=volt,
                y0=0,
                x1=volt,
                y1=1,
                line=dict(color='#FFFFFF', width=2, dash='solid'),
                name='vertical_line'
            )
                
        #category_orders={"": ["10 mV/s", "20 mV/s"]}
        show_fig(fig2_1, **fig_props)

        img_cdl_1 = export_fig(fig2_1, **fig_props)
        btn = download_fig(img_cdl_1)

        if len(options)==6:

            fig2_2 = go.Figure()

            reg_map = v_map_cdl[["Velocity (mV/s)"]]*(10**-3)
            reg_map["(Ia-Ic)/2"] = np.nan

            for name, df in df_cdl.items():
                if name in options:
                    df["Potential (V)"] = df["Potential (V)"].round(3)
                    val = df.loc[df['Potential (V)'] == volt, 'Current (A)']
                    i_A, i_C = val[val > 0].mean(), val[val < 0].mean()
                    reg_map.loc[name, "(Ia-Ic)/2"] = (i_A-i_C)/2

            st.divider()

            fig2_2 = px.scatter(reg_map, x='Velocity (mV/s)', y='(Ia-Ic)/2', trendline='ols', trendline_color_override='#1f77b4')
            fig2_2.data[0].marker.color = '#1f77b4'
            show_fig(fig2_2, title='Cyclic Voltammetry: Double-Layer Capacitance Estimation',
                      x_label='V/s (A)', y_label=r'(I<sub>A</sub>-I<sub>C</sub>)/2 (A)')

            img_cdl_2 = export_fig(fig2_2, title='Cyclic Voltammetry: Double-Layer Capacitance Estimation',
                      x_label='V/s (A)', y_label=r'(I<sub>A</sub>-I<sub>C</sub>)/2 (A)', width=4)
            btn = download_fig(img_cdl_2)

            st.divider()

            results = px.get_trendline_results(fig2_2)
            coefficients = results.iloc[0]['px_fit_results'].params

            cdl = coefficients[1]
            cdl_area = cdl/(ele_area_cdl*0.01) 

            st.markdown(f"$Cdl = {cdl*1e6:.2f}\,uF$")
            st.markdown(f"$Cdl_{{area}} = {cdl_area*1e6:.2f}\,uF/cm2$")


# CSC
with tabs[2]:

    st.markdown("**Calculate charge-storage capacity from cyclic voltammetry (CV)**")
    df_csc = {}
    
    with st.expander("Import data file"):
        try:
            uploaded_files_csc = st.file_uploader("Choose a CSV or TXT file", type=['csv', 'txt'], 
                                                accept_multiple_files=True, key='csc_upload')
            for uploaded_file in uploaded_files_csc:
                df_csc.update({uploaded_file.name[:-4] : pd.read_table(uploaded_file, sep='\t', 
                                                                    index_col=None, header=0, usecols=range(2))})
            for name, df in df_csc.items():
                    df = df[['Potential (V)', 'Current (A)']]
        except:
            df_csc = {}
            st.error('Imported file does not match the calculation method') 

    if df_csc:

        for name, df in df_csc.items():
            min_U = df['Potential (V)'].min()
            for index, row in df.iterrows():
                df_csc[name] = df_csc[name].drop(index)
                if row['Potential (V)'] == min_U:
                    break
            df = df.reset_index(drop=True)

        fig3_1 = go.Figure()

        options = data_to_plot(df_csc.keys(), 'opt_csc')

        if len(options)==2:
            v_map = pd.DataFrame(index=df_csc.keys(), columns=["CSC (C)", "Velocity (mV/s)"])

            for name, df in df_csc.items():
                v_map["CSC (C)"][name] = integrate.trapz(df['Current (A)'], x=df['Potential (V)'])
                
            v_map = v_map.sort_values(by=["CSC (C)"], ascending=True)
            v_map["Velocity (mV/s)"] = [50, 200]

            with st.expander("Assigned CV velocity"):
                st.write(v_map.style.format({"CSC (C)" : "{:e}"}))

        with st.expander("Figure properties"):
            fig_props = {
                'title': st.text_input("**Title**", value="Cyclic voltammetry", key='title_csc'),
                'x_label': st.text_input("**X-axis label**", value=r'E vs. Ag/AgCl<sub>3</sub> (V)', key='xlabel_csc'),
                'y_label': st.text_input("**Y-axis label**", value=r'J (A⋅cm<sup>-2</sup>)', key='ylabel_csc'),
                'legend': True if len(options)>1 else False,
                'width': 4
            }

        ele_area_csc = get_ele_area("ele_csc")

        for name, df in df_csc.items():
            if name in options:
                label = f'{v_map["Velocity (mV/s)"][name]} mV/s' if len(options)==2 else name
                fig3_1.add_trace(go.Scatter(x=df['Potential (V)'], y=df['Current (A)']/(ele_area_csc*0.01), 
                                        name=label))
                   
        show_fig(fig3_1, **fig_props)

        img_csc_1 = export_fig(fig3_1, **fig_props)
        btn = download_fig(img_csc_1)

        st.divider()

        CV_cycles = 3

        if len(options)==2:

            csc_50 = ((v_map.loc[v_map["Velocity (mV/s)"] == 50, "CSC (C)"].iloc[0])/CV_cycles)/0.05
            csc_200 = ((v_map.loc[v_map["Velocity (mV/s)"] == 200, "CSC (C)"].iloc[0])/CV_cycles)/0.2
            csc_50_area = (csc_50)/(ele_area_csc*1e-2) #C/cm2
            csc_200_area = (csc_200)/(ele_area_csc*1e-2) #C/cm2

            col1, col2 = st.columns(2)
            with col1:        
                st.markdown(f"$CSC_{{50mV/s}} = {csc_50*1e3:.2f}\,mC$")
                st.markdown(f"$CSC_{{200mV/s}} = {csc_200*1e3:.2f}\,mC$")
            with col2:        
                st.markdown(f"$CSC_{{50mV/s, \, area}} = {csc_50_area*1e3:.2f}\,mC/cm^{2}$")
                st.markdown(f"$CSC_{{200mV/s, \, area}} = {csc_200_area*1e3:.2f}\,mC/cm^{2}$")

            
#CIL
with tabs[3]:

    st.markdown("**Charge injection limit (CIL)**")
    df_cil = {}

    with st.expander("Import data file"):
        try:
            uploaded_files_cil = st.file_uploader("Choose a CSV or TXT file", type=['csv', 'txt'], 
                                                accept_multiple_files=True, key='cil_upload')
            for uploaded_file in uploaded_files_cil:
                df_cil.update({uploaded_file.name[:-4] : pd.read_table(uploaded_file, sep='\t', 
                                                                    index_col=None, header=0, usecols=range(2))})
            for name, df in df_cil.items():
                    df = df[['Potential (V)', 'Elapsed Time (s)']]
        except:
            df_cil = {}
            st.error('Imported file does not match the calculation method')

    start, stop = 0.049, 0.054

    if df_cil:

        for name, df in df_cil.items():
            for index, row in df.iterrows():
                if row['Elapsed Time (s)'] < start or row['Elapsed Time (s)'] > stop:
                    df_cil[name] = df_cil[name].drop(index)
                else:
                    continue
            df = df.reset_index(drop=True)

        fig4_1 = go.Figure()

        options = data_to_plot(df_cil.keys(), 'opt_cil')

        if len(options)==17:
            a_map_cil = pd.DataFrame(index=df_cil.keys(), columns=['Amplitude', 'Current (mA)'])

            a, b = 0.05, 0.0508
            df_map = df_cil.copy()

            for name, df in df_map.items():
                for index, row in df.iterrows():
                    if row['Elapsed Time (s)'] < a or row['Elapsed Time (s)'] > b:
                        df_map[name] = df_map[name].drop(index)
                    else:
                        continue
                df = df.reset_index(drop=True)

            for name, df in df_map.items():
                a_map_cil['Amplitude'][name] = integrate.trapz(df['Potential (V)'], x=df['Elapsed Time (s)'])

            a_map_cil = a_map_cil.sort_values(by=['Amplitude'], ascending=False)
            a_map_cil['Current (mA)'] = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10, 11, 12]

            with st.expander("Assigned amplitude"):
                st.write(a_map_cil['Current (mA)'])

        with st.expander("Figure properties"):
            fig_props = {
                'title': st.text_input("**Title**", value="Biphasic pulse", key='title_cil'),
                'x_label': st.text_input("**X-axis label**", value="Time (s)", key='xlabel_cil'),
                'y_label': st.text_input("**Y-axis label**", value="Potential (V)", key='ylabel_cil'),
                'legend': True if len(options)>1 else False,
                'width': 3,
                'marker': 10,
                'range': [-2.2, 1.5]
            }

        for name, df in df_cil.items():
            if name in options:
                label = f'{a_map_cil["Current (mA)"][name]} mA' if len(options)==17 else name
                fig4_1.add_trace(go.Scatter(x=df['Elapsed Time (s)'], y=df['Potential (V)'], 
                                            mode='lines+markers', name=label, line=dict(width=1)))
                
        show_fig(fig4_1, **fig_props)

        img_cil = export_fig(fig4_1, **fig_props)
        btn = download_fig(img_cil)

            
