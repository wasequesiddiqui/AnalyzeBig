import streamlit as st
import pandas as pd
import altair as alt

st.markdown("<h1 style='text-align: center;'>Data Visualizer App on StreamLit</h1>",unsafe_allow_html=True)
st.markdown("---",unsafe_allow_html=True)
files_names = list()
files = st.file_uploader("upload multiple files", type=["csv"], accept_multiple_files=True)

if files:
    for file in files:
        files_names.append(file.name)
    print(files_names)
    selected_files = st.multiselect("Select Files: ", options=files_names)
    if selected_files:
        option = st.radio("Select Entity Against Date ", options=['Price','Open','High','Low','Vol.','Change %'])
        if option!=None:
            for file in files:
                if file.name in selected_files:
                    chart_data = pd.read_csv(file)
                    chart_data['Change %'] = chart_data['Change %'].str.replace('%', '')
                    chart_data["Change %"] = chart_data["Change %"].astype(float).fillna(0.0)
                    chart_data["Color_Val"] = "Green"
                    chart_data.loc[(chart_data["Change %"]<0),"Color_Val"] = "Red"
                    c = (
                        alt.Chart(chart_data,title = "Displaying plot for " + option + " for " + file.name)
                        .mark_circle()
                        .encode(x="Date"
                                , y=option
                                , size="Change %"
                                , tooltip=["Open", "High", "Low"]
                                , color= alt.Color('Color_Val',legend=None)
                                )
                        ).interactive()
                    st.altair_chart(c)

