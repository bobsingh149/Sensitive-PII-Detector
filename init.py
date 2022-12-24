import streamlit as st
from methods import local_css

def init_app():
    # Remove whitespace from the top of the page and sidebar
    st.markdown("""
            <style>
                   .css-18e3th9 {
                        padding-top: 0rem;
                        padding-bottom: 10rem;
                        padding-left: 5rem;
                        padding-right: 5rem;
                    }
                   .css-1d391kg {
                        padding-top: 2.5rem;
                        padding-right: 1rem;
                        padding-bottom: 3.5rem;
                        padding-left: 1rem;
                    }
            </style>
            """, unsafe_allow_html=True)





    local_css("style/style.css")

    st.markdown("""
        <style>
        .big {
            font-size:50px !important;
        }
        .med {
            font-size:30px !important;
        }
        .sm {
           font-size:15px;
        }
        .center
        {
            text-align:center;
            color:black;

        }



        </style>
        """, unsafe_allow_html=True)
