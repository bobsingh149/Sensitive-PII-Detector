from streamlit_option_menu import option_menu
from items import *
from methods import *
from init import init_app
import numpy as np


st.set_page_config(page_title="PII Detection",  layout="wide")

init_app()

with st.sidebar:
    selected = option_menu(menu_title='Main Menu',

                           options=side_menu_items,
                           icons=side_menu_icons,

                           menu_icon='cast',
                           default_index=0,

                           )




if selected==side_menu_items[0]:

    cur = option_menu(menu_title='Dashboard',

                      options=nav_items,
                      icons=nav_icons,
                      menu_icon='cast',
                      default_index=0,
                      orientation='horizontal'

                      )

    st.markdown('---')

    if cur == nav_items[0]:  # Upload File
        file_upload()

    elif cur == nav_items[1]:  # Model size
        model_size()

    elif cur == nav_items[2]:  # Testing

        pii_detection()


    elif cur == nav_items[3]:  # Results
        st.title('Mask PII')
        mask_PII()

elif selected=='Analysis':
    plotbar()
