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

class PneumoniaChestXray_Model:
    def PneumoniaChestXray_Predict(self):
        def import_and_predict_chest(image_data, model):

            size = (150, 150)
            image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
            image = np.asarray(image)
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.

            img_reshape = img[np.newaxis, ...]
            prediction = model.predict(img_reshape)
            pred = np.argmax(prediction)

            return pred
        option = st.radio('',
                              ['Textual Explanation', 'Model', 'Visual Explanation'])
        st.write(
                '<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

        st.write(":heavy_minus_sign:" * 34)

        if option == 'Textual Explanation':
            st.header('OUTLINE')
            pneu_cx_desc = st.expander('Description')
            pneu_cx_desc.write('''
                                Pneumonia is an infection that inflames the air sacs in one or both lungs. 
                                The air sacs may fill with fluid or pus (purulent material), 
                                causing cough with phlegm or pus, fever, chills, and difficulty breathing.
                                 A variety of organisms, including bacteria, viruses and fungi, can cause pneumonia.
                            ''')
            pneu_cx_sym = st.expander('Symptoms')
            pneu_cx_sym.write('''
                                -   Chest pain when you breathe or cough
                                -   Confusion or changes in mental awareness (in adults age 65 and older)
                                -   Cough, which may produce phlegm
                                -   Fatigue
                                -   Fever, sweating and shaking chills
                                -   Lower than normal body temperature (in adults older than age 65 and people with weak immune systems)
                                -   Nausea, vomiting or diarrhea
                                -   Shortness of breath
                                ''')
            pneu_cx_causes = st.expander('Causes')
            pneu_cx_causes.write('''
                                **Community-acquired pneumonia**
                                Community-acquired pneumonia is the most common type of pneumonia. 
                                It occurs outside of hospitals or other health care facilities. It may be caused by:
                                -   Bacteria. 
                                -   Bacteria-like organisms.
                                -   Fungi.
                                -   Viruses, including COVID-19.

                                **Hospital-acquired pneumonia**
                                -   Some people catch pneumonia during a hospital stay for another illness. 
                                -   Hospital-acquired pneumonia can be serious because the bacteria causing it may be more resistant to antibiotics and because the people who get it are already sick. 
                                -   Occurs in people who live in long-term care facilities or who receive care in outpatient clinics, including kidney dialysis centers. Like hospital-acquired pneumonia.
                                -   Can be caused by bacteria that are more resistant to antibiotics.

                                **Aspiration pneumonia**
                                -   Aspiration pneumonia occurs when you inhale food, drink, vomit or saliva into your lungs. 
                                -   Aspiration is more likely if something disturbs your normal gag reflex, such as a brain injury or swallowing problem, or excessive use of alcohol or drugs.
                                ''')
            pneu_cx_pre = st.expander('Prevention')
            pneu_cx_pre.write('''
                                -   Get vaccinated. 
                                -   Make sure children get vaccinated.
                                -   Practice good hygiene. 
                                -   Don't smoke. 
                                -   Keep your immune system strong
                                ''')

        if option == 'Model':

            chestClasses = ['NORMAL', 'PNEUMONIA']
            chestModel = load_model('models/chest.h5')

            st.title("Welcome to PNEUMONIA CHEST XRAY CLassifier")
            st.header("Identify what's the Chest-XRAY result!")

            if st.checkbox("These are the classes of PNeumonia"):
                st.write(chestClasses)

            file = st.file_uploader("Please upload aCHEST XRAY Image", type=[
                                    "jpg", "png", "jpeg"])

            if file is None:
                st.info("Please upload an Image file")
            else:
                image = Image.open(file)
                st.image(image, use_column_width=True)
                predictions = import_and_predict_chest(image, chestModel)
                result = predictions
                st.write(result)
                if result == 0:
                    st.write("Prediction : Normal ")
                elif result == 1:
                    st.write("Prediction : PNEUMONIA FOUND")

            def predict_sample_img(file_path_selected):
                image = Image.open(file_path_selected)
                #st.image(image, use_column_width=True)
                predictions = import_and_predict_chest(image, chestModel)
                result = predictions
                st.write(result)
                if result == 0:
                    st.write("Prediction : Normal ")
                elif result == 1:
                    st.write("Prediction : PNEUMONIA FOUND")


            ## nested button with session states
            if "button_clicked" not in st.session_state:
                st.session_state.button_clicked=False                
            def callback():
                st.session_state.button_clicked=True
            if (st.button('Predict on Sample Images',on_click=callback) or st.session_state.button_clicked):
                dirs=['PNEUMONIA']
                classChoice=st.selectbox("Class:",dirs)

                if classChoice=='PNEUMONIA':
                    col1, col2, col3 = st.columns(3)

                    with col1:                           
                        st.image("images/sample_images/Chest/PNEUMONIA/person1_virus_13.jpeg")
                        if st.button('Predict on this image',key="1"):
                            file_path_selected='images/sample_images/Chest/PNEUMONIA/person1_virus_13.jpeg'
                            predict_sample_img(file_path_selected)
                            
                    with col2:              
                        st.image("images/sample_images/Chest/PNEUMONIA/person2_bacteria_3.jpeg")
                        if st.button('Predict on this image',key="2"):
                            file_path_selected='images/sample_images/Chest/PNEUMONIA/person2_bacteria_3.jpeg'
                            predict_sample_img(file_path_selected)
                        
                    with col3:
                        st.image("images/sample_images/Chest/PNEUMONIA/person32_virus_71.jpeg")
                        if st.button('Predict on this image',key="13"):
                            file_path_selected='images/sample_images/Chest/PNEUMONIA/person32_virus_71.jpeg'
                            predict_sample_img(file_path_selected)


        if option == 'Visual Explanation':
            st_player('https://www.youtube.com/watch?v=EEup71O8I0E')