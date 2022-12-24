side_menu_items = ['Sensitive PII Detector',  'Analysis']
side_menu_icons = ['search', 'bar-chart']
side_menu_style = {"container": {"padding": "0!important", "background-color": "#fafafa"},
                   "icon": {"color": "orange", "font-size": "25px"},
                   "nav-link": {"font-size": "25px", "text-align": "left", "margin": "0px",
                                "--hover-color": "#eee"},
                   "nav-link-selected": {"background-color": "green"}, }

nav_items = ['Datasets', 'Train Model', 'Detect PII','Mask PII']
nav_icons = ['upload', 'table', 'pie-chart']
nav_style = {
    "body": {'color': 'black'},
    "container": {"padding": "0!important", "background-color": "red"},
    "icon": {"color": "orange", "font-size": "25px"},

    "nav-link": {"font-size": "25px", "text-align": "left", "margin": "0px",
                 "--hover-color": "red", "background-color": "black"},
    "nav-link-selected": {"background-color": "grey"}, }

colors = {'B-per': "#FF0000", 'I-per': "#FF0000", 'B-geo': '#c719fa', 'I-geo': '#c719fa',
          'B-org': '#dab165', 'I-org': '#dab165', 'B-gpe': '#4fe1d4', 'I-gpe': '#4fe1d4',
          'B-tim': '#D5F9DE', 'I-tim': '#D5F9DE'}

displacy_options = {"ents": ['B-per', 'I-per', 'B-geo', 'I-geo', 'B-org', 'I-org', 'B-gpe', 'I-gpe', 'B-tim', 'I-tim'],
           "colors": colors}


all_ents=[
"CREDIT_CARD",
"CRYPTO",
"DATE_TIME",
"EMAIL_ADDRESS",
"IBAN_CODE",
"IP_ADDRESS",
"NRP",
"LOCATION",
"PERSON",
"PHONE_NUMBER",
"MEDICAL_LICENSE",
"URL",
]


