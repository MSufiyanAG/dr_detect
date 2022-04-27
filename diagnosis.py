import streamlit as st
import streamlit.components.v1 as components
from streamlit_player import st_player
import numpy as np
from bokeh.models.widgets import Div
from PIL import Image, ImageOps
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

from models_code import DR,SC,PCX,BTM,BLT,KOX

DR_code=DR.Dr_Model()
SC_code=SC.Sc_Model()
PCX_code=PCX.Pcx_Model()
BTM_code=BTM.Btm_Model()
BLT_code=BLT.Blt_Model()
KOX_code=KOX.Kox_Model()

class model_predict:
    def predict(self):
        diseases = ["Select", "Diabetic-Retinopathy", "Skin-Cancer", "Brain-Tumor-MRI",
                    "Pneumonia-Chest-XRAY", "Breast-Lesions-Tumor-Cancer", "Knee-Osteoarthritis-XRAY"]

        st.header("Diseases")
        diagnose = st.selectbox("", diseases)

        if diagnose == 'Select':

            st.header("PREFACE")
            st.write('''
                    The topics included for each disease in this section are:

                    -   **Textual Explanation** : It contains a brief description about the disease along with its symptoms , causes, preventions/treatment.
                    -   **Model** : The Deep Learning based solution proposed for the specific disease .
                    -   **Visual Explanation** : A YouTube video about the condition of disease.
                    ''')

            col1, col2, col3 = st.columns([6, 6, 6])
            with col1:
                st.write("")
            with col2:
                st.image('images/web_images/earth-small.gif')
            with col3:
                st.write("")
            # st.image('img/earth-small.gif')

        ###########################
        ## Diabetic-Retionapathy ##
        ###########################

        if diagnose == 'Diabetic-Retinopathy':

            DR_code.DrPredict()
        #################
        ## Skin-Cancer ##
        #################

        if diagnose == "Skin-Cancer":

            SC_code.ScpPredict()

        ###########################
        ## Pneumonia-Chest-XRAY ###
        ###########################

        if diagnose == "Pneumonia-Chest-XRAY":

           PCX_code.PcxPredict()

        #####################
        ## Brain-Tumor-MRI ##
        #####################

        if diagnose == "Brain-Tumor-MRI":

            BTM_code.BtmPredict()

        #################################
        ## Breast-Lesions-Tumor-Cancer ##
        #################################

        if diagnose == 'Breast-Lesions-Tumor-Cancer':

            BLT_code.BltPredict()

        ##############################
        ## Knee-Osteoarthritis-XRAY ##
        ##############################

        if diagnose == 'Knee-Osteoarthritis-XRAY':

            KOX_code.KoxPredict()