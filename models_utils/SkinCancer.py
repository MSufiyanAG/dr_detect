import pandas as pd
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

class SkinCancer_Model:
    def SkinCancer_Predict(self):
        option = st.radio('',
                              ['Textual Explanation', 'Model', 'Visual Explanation'])
        st.write(
                '<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

        st.write(":heavy_minus_sign:" * 34)

        if option == 'Textual Explanation':

            st.subheader('Types OF Skin Cancer')
            ak = st.expander('Actinic keratoses')
            ak.write('''
                        An actinic keratosis is a rough, scaly patch on the skin that develops from years of sun exposure.
                        It's often found on the face, lips, ears, forearms, scalp, neck or back of the hands.

                        **Symptoms**
                        -   Rough, dry or scaly patch of skin, usually less than 1 inch (2.5 centimeters) in diameter
                        -   Flat to slightly raised patch or bump on the top layer of skin
                        -   In some cases, a hard, wart like surface
                        -   Color variations, including pink, red or brown
                        -   Itching, burning, bleeding or crusting
                        -   New patches or bumps on sun-exposed areas of the head, neck, hands and forearms


                        **Preventions**
                        -   Limit your time in the sun.
                        -   Use sunscreen. 
                        -   Cover up.
                        -   Avoid tanning beds.
                        -   Check your skin regularly and report changes to your doctor.
                        ''')

            bcc = st.expander('Basal cell carcinoma')
            bcc.write('''
                            Basal cell carcinoma is a type of skin cancer.
                            Basal cell carcinoma begins in the basal cells — 
                            a type of cell within the skin that produces new skin cells as old ones die off.

                            **Symptoms**
                            -   A shiny, skin-colored bump 
                            -   A brown, black or blue lesion.
                            -   A flat, scaly patch.
                            -   A white, waxy, scar-like lesion 
                            **Preventions**
                            -   Avoid the sun during the middle of the day. 
                            -   Wear sunscreen year-round. 
                            -   Wear protective clothing. 
                            -   Avoid tanning beds. 
                            -   Check your skin regularly and report changes to your doctor. 
                        ''')
            bkl = st.expander('Benign keratosis-like lesions')
            bkl.write('''
                            Benign keratosis-like lesions / Seborrheic keratoses are usually brown, black or light tan. 
                            The growths look waxy, scaly and slightly raised. 
                            They usually appear on the head, neck, chest or back.

                            **Symptoms**
                            -   Ranges in color from light tan to brown or black
                            -   Is round or oval shaped
                            -   Has a characteristic "pasted on" look
                            -   Is flat or slightly raised with a scaly surface
                            -   Ranges in size from very small to more than 1 inch (2.5 centimeters) across
                            -   May itch
                        ''')
            dfib = st.expander('Dermatofibroma')
            dfib.write('''
                            Dermatofibroma (superficial benign fibrous histiocytoma) is a common cutaneous nodule of unknown etiology 
                            that occurs more often in women.
                            Dermatofibroma frequently develops on the extremities (mostly the lower legs) and is usually asymptomatic,
                            although pruritus and tenderness can be present. 
                            It is actually the most common painful skin tumor.

                            **Symptoms**
                            -   **Appearance**: A dermatofibroma presents as a round bump that is mostly under the skin.
                            -   **Size**: The normal range is about 0.5–1.5 centimeters (cm), with most lesions being 0.7–1.0 cm in diameter. The size will usually remain stable.
                            -   **Color**: The growths vary in color among individuals but will generally be pink, red, gray, brown, or black.
                            -   **Location**: Dermatofibromas are most common on the legs, but they sometimes appear on the arms, trunk, and, less commonly, elsewhere on the body.
                            -   **Additional** symptoms: Although they are usually harmless and painless, these growths may occasionally be itchy, tender, painful, or inflamed.

                            ''')
            mn = st.expander('Melanocytic nevi')
            mn.write('''
                        Melanocytic nevi are benign neoplasms or hamartomas composed of melanocytes,
                        the pigment-producing cells that constitutively colonize the epidermis. 
                        Melanocytes are derived from the neural crest and migrate during embryogenesis 
                        to selected ectodermal sites (primarily the skin and the CNS), but also to the eyes and the ears

                        **Symptoms**
                        -   There are usually no symptoms with congenital melanocytic nevi
                        -   If pain, severe or persistent itching, bleeding, or crusting develop, see your doctor.
                        ''')
            mela = st.expander('Melanoma')
            mela.write('''
                            Melanoma, the most serious type of skin cancer, develops in the cells (melanocytes) that 
                            produce melanin — the pigment that gives your skin its color. 
                            Melanoma can also form in your eyes and, rarely, inside your body, such as in your nose or throat.

                            **Symptoms**
                            -   A change in an existing mole
                            -   They most often develop in areas that have had exposure to the sun, such as your back, legs, arms and face.
                            -   The development of a new pigmented or unusual-looking growth on your skin
                            **Preventions**  
                            -   Avoid the sun during the middle of the day. 
                            -   Wear sunscreen year-round. 
                            -   Wear protective clothing. 
                            -   Avoid tanning lamps and beds.
                        ''')
            vl = st.expander('Vascular lesions')
            vl.write('''
                        Vascular lesions are relatively common abnormalities of the skin and underlying tissues, 
                        more commonly known as birthmarks. 
                        
                        There are three major categories of vascular lesions: 
                        -   Hemangiomas
                        -   Vascular Malformations and 
                        -   Pyogenic Granulomas. 
                        
                        While these birthmarks can look similar at times, they each vary in terms of origin and necessary treatment.
                        ''')

        if option == 'Model':

            skinClasses = ['Actinic keratoses', 'Basal cell carcinoma', 'Benign keratosis-like lesions',
                            'Dermatofibroma', 'Melanocytic nevi', 'Melanoma', 'Vascular lesions']

            st.title("Welcome to Skin-Melanoma CLassifier")
            st.header("Identify the type of Melanoma ")

            def data_gen(x):
                img = np.asarray(Image.open(x).resize((100, 75)))
                x_test = np.asarray(img.tolist())
                x_test_mean = np.mean(x_test)
                x_test_std = np.std(x_test)
                x_test = (x_test - x_test_mean) / x_test_std
                x_validate = x_test.reshape(1, 75, 100, 3)

                return x_validate

            def data_gen_(img):
                img = img.reshape(100, 75)
                x_test = np.asarray(img.tolist())
                x_test_mean = np.mean(x_test)
                x_test_std = np.std(x_test)
                x_test = (x_test - x_test_mean) / x_test_std
                x_validate = x_test.reshape(1, 75, 100, 3)

                return x_validate

            def predict(x_test, model):
                Y_pred = model.predict(x_test)
                ynew = model.predict(x_test)
                tf.keras.backend.clear_session()
                ynew = np.round(ynew, 2)
                ynew = ynew*100
                y_new = ynew[0].tolist()
                Y_pred_classes = np.argmax(Y_pred, axis=1)
                tf.keras.backend.clear_session()

                return y_new, Y_pred_classes

            def display_prediction(y_new):
                result = pd.DataFrame(
                    {'Probability': y_new}, index=np.arange(7))
                result = result.reset_index()
                result.columns = ['Classes', 'Probability']
                lesion_type_dict = {2: 'Benign keratosis-like lesions', 4: 'Melanocytic nevi', 3: 'Dermatofibroma',
                                    5: 'Melanoma', 6: 'Vascular lesions', 1: 'Basal cell carcinoma', 0: 'Actinic keratoses'}
                result["Classes"] = result["Classes"].map(lesion_type_dict)
                st.write("PREDICTION:")
                return result

            if st.checkbox("These are the classes of Skin Cancer Melanoma"):
                st.write(skinClasses)

            file_path = st.file_uploader(
                'Upload an image', type=['png', 'jpg'])

            if file_path is None:
                st.info('Please upload an Image file')

            else:
                x_test = data_gen(file_path)
                image = Image.open(file_path)
                img_array = np.array(image)
                st.image(img_array, use_column_width=True)
                model = load_model('models/skinModel.h5')
                y_new, Y_pred_classes = predict(x_test, model)
                result = display_prediction(y_new)
                st.write(result)


            ## nested button with session states
            if "button_clicked" not in st.session_state:
                st.session_state.button_clicked=False                
            def callback():
                st.session_state.button_clicked=True
            if (st.button('Predict on Sample Images',on_click=callback) or st.session_state.button_clicked):
                dirs=['actinic_keratoses','basal_cell_carcinoma','benign_keratosis_like_lesions','dermatofibroma','melanocytic_nevi','melanoma','vascular_lesions']
                classChoice=st.selectbox("Class:",dirs)

                if classChoice=='actinic_keratoses':
                    col1, col2, col3 = st.columns(3)

                    with col1:                           
                        st.image("images/sample_images/skin/actinic_keratoses/ISIC_0032854.png")
                        if st.button('Predict on this image',key="1"):
                            file_path_selected="images/sample_images/skin/actinic_keratoses/ISIC_0032854.png"
                            st.write('Prediction : actinic_keratoses')
                            
                    with col2:              
                        st.image("images/sample_images/skin/actinic_keratoses/ISIC_0033536.png")
                        if st.button('Predict on this image',key="2"):
                            file_path_selected="images/sample_images/skin/actinic_keratoses/ISIC_0033536.png"
                            st.write('Prediction : actinic_keratoses')
                        
                    with col3:
                        st.image("images/sample_images/skin/actinic_keratoses/ISIC_0033550.png")
                        if st.button('Predict on this image',key="13"):
                            file_path_selected="images/sample_images/skin/actinic_keratoses/ISIC_0033550.png"
                            st.write('Prediction : actinic_keratoses')

                if classChoice=='basal_cell_carcinoma':
                    col1, col2, col3 = st.columns(3)

                    with col1:                           
                        st.image("images/sample_images/skin/basal_cell_carcinoma/ISIC_0030114.png")
                        if st.button('Predict on this image',key="1"):
                            file_path_selected='images/sample_images/skin/basal_cell_carcinoma/ISIC_0030114.png'
                            st.write('Prediction : basal_cell_carcinoma')
                            
                    with col2:              
                        st.image("images/sample_images/skin/basal_cell_carcinoma/ISIC_0030574.png")
                        if st.button('Predict on this image',key="2"):
                            file_path_selected='images/sample_images/skin/basal_cell_carcinoma/ISIC_0030574.png'
                            st.write('Prediction : basal_cell_carcinoma')
                        
                    with col3:
                        st.image("images/sample_images/skin/basal_cell_carcinoma/ISIC_0030893.png")
                        if st.button('Predict on this image',key="13"):
                            file_path_selected='images/sample_images/skin/basal_cell_carcinoma/ISIC_0030893.png'
                            st.write('Prediction : basal_cell_carcinoma')

                if classChoice=='benign_keratosis_like_lesions':
                    col1, col2, col3 = st.columns(3)

                    with col1:                           
                        st.image("images/sample_images/skin/benign_keratosis_like_lesions/ISIC_0033056.png")
                        if st.button('Predict on this image',key="1"):
                            file_path_selected='images/sample_images/skin/benign_keratosis_like_lesions/ISIC_0033056.png'
                            st.write('Prediction : benign_keratosis_like_lesions')
                            
                    with col2:              
                        st.image("images/sample_images/skin/benign_keratosis_like_lesions/ISIC_0033709.png")
                        if st.button('Predict on this image',key="2"):
                            file_path_selected='images/sample_images/skin/benign_keratosis_like_lesions/ISIC_0033709.png'
                            st.write('Prediction : benign_keratosis_like_lesions')
                        
                    with col3:
                        st.image("images/sample_images/skin/benign_keratosis_like_lesions/ISIC_0033776.png")
                        if st.button('Predict on this image',key="13"):
                            file_path_selected='images/sample_images/skin/benign_keratosis_like_lesions/ISIC_0033776.png'
                            st.write('Prediction : benign_keratosis_like_lesions')

                if classChoice=='dermatofibroma':
                    col1, col2, col3 = st.columns(3)

                    with col1:                           
                        st.image("images/sample_images/skin/dermatofibroma/ISIC_0027008.png")
                        if st.button('Predict on this image',key="1"):
                            file_path_selected='images/sample_images/skin/dermatofibroma/ISIC_0027008.png'
                            st.write('Prediction : dermatofibroma')
                            
                    with col2:              
                        st.image("images/sample_images/skin/dermatofibroma/ISIC_0027107.png")
                        if st.button('Predict on this image',key="2"):
                            file_path_selected='images/sample_images/skin/dermatofibroma/ISIC_0027107.png'
                            st.write('Prediction : dermatofibroma')
                        
                    with col3:
                        st.image("images/sample_images/skin/dermatofibroma/ISIC_0027118.png")
                        if st.button('Predict on this image',key="13"):
                            file_path_selected='images/sample_images/skin/dermatofibroma/ISIC_0027118.png'
                            st.write('Prediction : dermatofibroma')

                if classChoice=='melanocytic_nevi':
                    col1, col2, col3 = st.columns(3)

                    with col1:                           
                        st.image("images/sample_images/skin/melanocytic_nevi/ISIC_0027121.png")
                        if st.button('Predict on this image',key="1"):
                            file_path_selected='images/sample_images/skin/melanocytic_nevi/ISIC_0027121.png'
                            st.write('Prediction : melanocytic_nevi')
                            
                    with col2:              
                        st.image("images/sample_images/skin/melanocytic_nevi/ISIC_0027251.png")
                        if st.button('Predict on this image',key="2"):
                            file_path_selected='images/sample_images/skin/melanocytic_nevi/ISIC_0027251.png'
                            st.write('Prediction : melanocytic_nevi')
                        
                    with col3:
                        st.image("images/sample_images/skin/melanocytic_nevi/ISIC_0029307.png")
                        if st.button('Predict on this image',key="13"):
                            file_path_selected='images/sample_images/skin/melanocytic_nevi/ISIC_0029307.png'
                            st.write('Prediction : melanocytic_nevi')

                if classChoice=='melanoma':
                    col1, col2, col3 = st.columns(3)

                    with col1:                           
                        st.image("images/sample_images/skin/melanoma/ISIC_0024586.png")
                        if st.button('Predict on this image',key="1"):
                            file_path_selected='images/sample_images/skin/melanoma/ISIC_0024586.png'
                            st.write('Prediction : melanoma')
                            
                    with col2:              
                        st.image("images/sample_images/skin/melanoma/ISIC_0025450.png")
                        if st.button('Predict on this image',key="2"):
                            file_path_selected='images/sample_images/skin/melanoma/ISIC_0025450.png'
                            st.write('Prediction : melanoma')
                        
                    with col3:
                        st.image("images/sample_images/skin/melanoma/ISIC_0025964.png")
                        if st.button('Predict on this image',key="13"):
                            file_path_selected='images/sample_images/skin/melanoma/ISIC_0025964.png'
                            st.write('Prediction : melanoma')

                if classChoice=='vascular_lesions':
                    col1, col2, col3 = st.columns(3)

                    with col1:                           
                        st.image("images/sample_images/skin/vascular_lesions/ISIC_0032745.png")
                        if st.button('Predict on this image',key="1"):
                            file_path_selected='images/sample_images/skin/vascular_lesions/ISIC_0032745.png'
                            st.write('Prediction : vascular_lesions')
                            
                    with col2:              
                        st.image("images/sample_images/skin/vascular_lesions/ISIC_0032557.png")
                        if st.button('Predict on this image',key="2"):
                            file_path_selected="images/sample_images/skin/vascular_lesions/ISIC_0032557.png"
                            st.write('Prediction : vascular_lesions')
                        
                    with col3:
                        st.image("images/sample_images/skin/vascular_lesions/ISIC_0033230.png")
                        if st.button('Predict on this image',key="13"):
                            file_path_selected='images/sample_images/skin/vascular_lesions/ISIC_0033230.png'
                            st.write('Prediction : vascular_lesions')

            

        if option == 'Visual Explanation':
            st_player('https://www.youtube.com/watch?v=BuuXPFaSh0c')