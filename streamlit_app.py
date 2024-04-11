import streamlit as st
import base64
from PIL import Image
import opencv-python
import accumulated_weights_model as awm
import image_dehazer_model as idm
import dcp_model as dcpm

# Define a function to get the image as base64 using st.cache_data

def set_png_as_page_bg(png_file):
    # Open the image file and read its contents
    with open(png_file, "rb") as f:
        data = f.read()
        base64_img = base64.b64encode(data).decode()
        
    image_path = Image.open("error.jpg")
    st.sidebar.image(image_path, use_column_width=True)
    st.sidebar.header("Help Desk !")
    st.sidebar.text("Rescue Mode uses image_dehazer model \n with auto ambience light adjustment for faster \n refresh rate and better visibility")
    st.sidebar.text("User Mode runs dark channel prior model \n with object detection to improvise the visibility \n and detect objects simultaneously (comparatively slower)")
    st.sidebar.text("Advanced Mode uses accumulated weight \n model that avoids the capturing \n of regularly moving particles and allow \n the non-moving objects behind \n the heavy dense smoke particles")
    # st.sidebar.text("D ---> Accumulated Model")
    # st.sidebar.text("Y ---> Yolo Object Detection")
    # st.sidebar.text("T ---> Quit Yolo")
    # st.sidebar.text("I ---> Heat Vision")
    # st.sidebar.text("J ---> Quit heat Vision")
    # st.sidebar.text("Q ---> Quit Program")
    # controler = Image.open("Slide1.JPG")
    # st.sidebar.image(controler, use_column_width=True)

    # Set the background image using CSS and HTML
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url('data:image/png;base64,{base64_img}');
        background-size: cover;
    }}
    
    [data-testid="stHeader"]{{
    background-color: black;
    }}

    [data-testid="stButton"]{{
    height: 200px;
    width: 40px;
    padding: 2px;
    margin: 0px;
    background-color: rgba(0,0,0,0.5);
    }}

    [data-testid="baseButton-secondary"]{{
    height: 80px;
    width: 200px;
    font-size:100px;
    padding: 2px;
    margin: 0px;
    background: black;
    }}
    
    [data-testid="stSidebarUserContent"]{{
    padding: 0px;
    margin: 0px;
    }}

    [data-testid="element-container"]{{
    padding: 0px;
    margin: 0px;
    text-align: center;
    color: white;
    font-weight: bold;
    font-size: 100px;
    }}
      
    .st-emotion-cache-1629p8f.e1nzilvr2 h2 {{
    color: red;
    font-size: 25px;
    padding: 0px;
    font-weight: bold;
    }}

    [data-testid="stText"] {{
    color: white;
    padding-left: 5px;
    margin-left: 2px;
    }}
    
    [data-testid="StyledLinkIconContainer"] {{
        color: white;
        font-family: 'Roboto', sans-serif;
    }}
    
    [data-testid="stImage"] {{
        padding: 0px;
        margin: 0px;
        float: left;
    }}
    
    [data-testid="stSidebarNavItems"]{{
        padding: 0px;
        
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Call the function to set the background image
set_png_as_page_bg("purple smoke.jpeg.jpg")

# Rest of your Streamlit code goes here
st.title("De-HAZE DYNAMOS")

#buttons
st.title("Real-Time Dehazing/De-smoking")
Rescue = st.button("Rescue Mode")
User = st.button("User Mode")
Advanced = st.button("Advanced Mode")

    
#####################################################################################


if Rescue == True:
    idm.main()

elif User == True:
    dcpm.main()

elif Advanced == True:
    awm.main()
