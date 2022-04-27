import streamlit as st
import streamlit.components.v1 as components
from streamlit_player import st_player

class information():
    def print_data(self):
        col1, col2, col3 = st.columns([6, 15, 6])
        with col1:
            st.write("")
        with col2:
            st.image('images/web_images/LOGO.png')
        with col3:
            st.write("")

        st.title('About')
        st.write('''
                Early recognition of results of medical reports is the key to preventing risky complications (even death),
                even more so now that millions of COVID-19 patients require care. 
                Deep learning algorithms may be able to automatically detect abnormalities. 
                Once alerted, clinicians can take appropriate decisions to avoid life threatening complications
                ''')

        st.write('''
                Delay or Lack of proper diagnosis of various health-related issues is a major concern,
                so Early Detection of such reports is even more important as COVID-19 cases continue to surge,
                these steps can be time consuming and are still prone to human error,
                especially in stressful situations when hospitals are at capacity.
                ''')

        st.write('''
                Deep learning algorithms simplify complex data analysis,
                so abnormalities are determined and prioritized more precisely.
                The insights that Convolutional Neural Networks (CNNs) provide,
                help medical professionals to notice health issues of their patients on time and more accurately.
                ''')

        st.write('''
                The proposed solution will help to understand the result in the state of an emergency even by a paramedic/Jr Doctor in absence of the specialist. 
                ''')

        col1, col2, col3 = st.columns([6, 15, 6])
        with col1:
            st.write("")
        with col2:
            st.image('images/web_images/drs.png')
            st.write('--- NEVER FORGET TO CONSULT A SPECIALIST ---',)
        with col3:
            st.write("")

        # st.image('img/drs.png')