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

from models_utils import BrainTumorMRI, DiabeticRetinopathy,BreastLesions, KneeOsteoarthritisXray, PneumoniaChestXray, SkinCancer

DR=DiabeticRetinopathy.DiabeticRetinopathy_Model()
SC=SkinCancer.SkinCancer_Model()
PCX=PneumoniaChestXray.PneumoniaChestXray_Model()
BTM=BrainTumorMRI.BrainTumorMRI_Model()
BLT=BreastLesions.BreastLesions_Model()
KOX=KneeOsteoarthritisXray.KneeOsteoarthritisXray_Model()

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

            DR.DiabeticRetinopathy_Predict()
        #################
        ## Skin-Cancer ##
        #################

        if diagnose == "Skin-Cancer":

            SC.SkinCancer_Predict()

        ###########################
        ## Pneumonia-Chest-XRAY ###
        ###########################

        if diagnose == "Pneumonia-Chest-XRAY":

           PCX.PneumoniaChestXray_Predict()

        #####################
        ## Brain-Tumor-MRI ##
        #####################

        if diagnose == "Brain-Tumor-MRI":

            BTM.BrainTumorMRI_Predict()

        #################################
        ## Breast-Lesions-Tumor-Cancer ##
        #################################

        if diagnose == 'Breast-Lesions-Tumor-Cancer':

            BLT.BreastLesions_Predict()

        ##############################
        ## Knee-Osteoarthritis-XRAY ##
        ##############################

        if diagnose == 'Knee-Osteoarthritis-XRAY':

            KOX.KneeOsteoarthritisXray_Predict()