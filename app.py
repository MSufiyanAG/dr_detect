import streamlit as st
import streamlit.components.v1 as components
import time
import base64
from pathlib import Path
from bokeh.models.widgets import Div

import requests
from io import BytesIO

from about import information
from diagnosis import model_predict
ia=information()
mp=model_predict()

st.set_page_config(page_title='DR.Detect', page_icon='images/web_images/LOGO.png')

#st.set_option('deprecation.showfileUploaderEncoding', False)
#st.set_option('deprecation.showPyplotGlobalUse', False)


#with open('style.css') as f:
    #st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def main():
    

    def img_to_bytes(img_path):
        img_bytes = Path(img_path).read_bytes()
        encoded = base64.b64encode(img_bytes).decode()
        return encoded
    page_bg_img = """
                    <style>
                        body {
                            color:#343148 ;
                            background-color:  #b2b7bb;
                        }
                        
                        stTextInput>div>div>input {
                            color: #ff0000;
                        }
                        
                        div.css-1aumxhk {
                        color: #011839;
                        background-image: none;
                        color: #ffffff
                        }
                        </style>
                    """
    st.write(
            '''<style>
                        body {
                            color:#343148 ;
                            background-color:  #b2b7bb;
                        }
                        
                        stTextInput>div>div>input {
                            color: #ff0000;
                        }
                        
                        div.css-1aumxhk {
                        color: #011839;
                        background-image: none;
                        color: #ffffff
                        }
                        </style>''', unsafe_allow_html=True)                
    #st.markdown(page_bg_img, unsafe_allow_html=True)     
    # st.sidebar.title('DR.Detect')

    sidebar = ("https://media.giphy.com/media/l0IylQoMkcbZUbtKw/giphy.gif")
    sidebar_html = "<img  style= 'vertical-align: bottom' src='data:image/gif;base64,{}' class='img-fluid' width='300' height='400'>".format(
        img_to_bytes("images/web_images/dr.gif")
    )
    st.sidebar.markdown(
        sidebar_html, unsafe_allow_html=True,
    )

    activities = ["[-About-]", "[-Diagnosis-]"]
    choice = st.sidebar.selectbox("", activities)

    st.sidebar.write("----------------------------")

    st.sidebar.title('***CONTRIBUTORS***')
    st.sidebar.write('''
                    **Syed Mahboob Abrar Ali**                    
                    1604-18-733-088


                    **Mohd Abdul Azeem**                         
                    1604-18-733-089


                    **Mohammed Sufiyan Abdullah Ghori**          
                    1604-18-733-094

                ''')

    st.sidebar.write("----------------------------")
    st.sidebar.write(f'''
    <a target="_self" href="https://github.com/MSufiyanAG/dr_detect">
        <button>
            GitHub
        </button>
    </a>
    ''',
    unsafe_allow_html=True
)

    if choice == "[-About-]":

        ia.print_data()

    if choice == "[-Diagnosis-]":

        mp.predict()


if __name__ == '__main__':
    main()
