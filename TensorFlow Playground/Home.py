import streamlit as st

custom_css = """
    <style>
        body {
            background-color: #f0f0f0; /* Specify your desired background color */
        }
    </style>
"""
# Add the custom CSS to the Streamlit app
st.markdown(custom_css, unsafe_allow_html=True)


styled_text = """
    <div style="font-size: 50px; color: #FF5733; text-align: center;">
        <span style="font-weight: bold;">WELCOME TO</span> 
    </div>
    <div style="font-size: 50px; color: #FF5733; text-align: left;">
        <span style="font-weight: bold;">TENSORFLOW PLAYGROUND</span> 
    </div>
    <div style="font-size: 25px; color: #3366FF; text-align: right;">
        <span style="font-weight: bold;">D.L. Narasimha Rao</span>
    </div>
"""

# Display the styled text using Markdown
st.markdown(styled_text, unsafe_allow_html=True)

image = 'tensor.jpg'
st.image(image,use_column_width=True)