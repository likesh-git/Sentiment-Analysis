#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# importing libraries

import pandas as pd
import base64
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pickle
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px
import webbrowser
from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt


# In[ ]:


# declaring global variables
global app
app=dash.Dash(external_stylesheets=[dbc.themes.SUPERHERO],suppress_callback_exceptions=True)
global stopwords
global filename
global count
global df
global image_filename
global encoded_image


# In[ ]:





# In[ ]:


# defing function

def load_model():
    
    global scrappedReviews
    scrappedReviews=pd.read_csv('scrappedReviews.csv')
    
    global recreated_model
    file=open('pickle_model.pkl','rb')
    recreated_model=pickle.load(file)
    
    global vocab
    vocab=pickle.load(open('features.pkl','rb'))
    
    return 


def check_review(reviewText):
    #reviewText has to be vectorised, that vectorizer is not saved yet
    #load the vectorize and call transform and then pass that to model preidctor
    #load it later
    load_model()
    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace",vocabulary=vocab)
    vectorised_review = transformer.fit_transform(loaded_vec.fit_transform([reviewText]))


    # Add code to test the sentiment of using both the model
    # 0 == negative   1 == positive
    
    return recreated_model.predict(vectorised_review)


def open_browser():
    
    webbrowser.open_new('http://127.0.0.1:8050/')
    
    return 


def mywordcloud(data,title=None):
    wordcloud=WordCloud(
        width=200,
        height=100,
        background_color='yellow',
        

        stopwords=stopwords,
        max_words=300,
        max_font_size=40,
        scale=3,
        random_state=1).generate(str(data))
    

    
    wordcloud.to_file(filename)

    
    return filename


















def create_app_ui():

    main_layout=html.Div(
        
        [
            html.H1(id='main_title',children='sentiment analysis  with insights',style={"text-align":"center"},
                    className = 'display-3 mb-4'),            
            
            html.Div([
                
                dcc.Graph(figure=px.pie(count, names='positivity'),className='positive reviews v/s negative reviews ')
                ]),
            
            html.Br(),
            html.H1(id='word_cloud_title', children='MOST USED WORDS',style={"text-align":"center"}),
            html.Br(),
            
           dbc.Col( html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()),style={"text-align":"center"})
                   ,style={"text-align":"center"}),
            html.Br(),
           
            dbc.Col(dcc.Dropdown(
                
                
                id='dropdown',
                placeholder = 'Select a Review',
                options=[{'label': i[:100] + "...", 'value': i} for i in df.reviews],
                value = df.reviews[2],
                style = {'margin-bottom': '30px'}
            ),
                   width={'size':10, 'offset':1}),
                    
            dbc.Col(dbc.Button(
                className="mt-2 mb-3",
                id='button_review',
                children='check',
                color='dark',
                #style={'width': '100px', 'height': 40,"text-align":"center"}
                style = {'background-color': '#e63946ff','width': '220px',"text-align":"center"}
            ),width={'size':10, 'offset':5}),
            
            html.H3(id='result',style={"text-align":"center"}),
            
            
            dbc.Col(dcc.Textarea(

                    id='text_area_review',
                    placeholder='enter the review here....',
                    style={'width':'100%', 'height': 100}
                ),
                    width={'size':10, 'offset':1}),
            html.Br(),
           dbc.Col(dbc.Button(
                id='button_revieww',
                children='check',
                color='dark',
                style = {'background-color': '#e63946ff','width': '220px'}
            ),width={'size':10, 'offset':5}),
            
            html.Br(),
            
            dbc.Col(html.H3(id='result1'),style={"text-align":"center"})
        ]
    )
    
    
    return main_layout

@app.callback(
    Output('result', 'children'),
    [
        Input('button_review', 'n_clicks')
    ],
    [
     State('dropdown', 'value')
     ]
    )
def update_app_ui(n_clicks,textarea_value):
    if(n_clicks>0):
        
        print('data type of ', str(type(textarea_value)))
        print('value = ', str(textarea_value))

        response=check_review(textarea_value)

        if (response[0]==0):
            result1=dbc.Alert("It is a negative review", color="danger")

        elif(response[0]==1):
            result1=dbc.Alert("It is a positive review", color="success")

        else:
            result1='unknown '  
            
    else:
        result='result of dropdown box'
    
    
    return result1



@app.callback(
    Output('result1', 'children'),
    [
        Input('button_revieww', 'n_clicks')
    ],
    [
     State('text_area_review', 'value')
     ]
)
def update_app_ui(n_clicks,textarea_value):
    if(n_clicks==0):
        return 
    print('data type of ', str(type(textarea_value)))
    print('value = ', str(textarea_value))
    
    response=check_review(textarea_value)
    
    if (response[0]==0):
        result1=result1=dbc.Alert("It is a negative review", color="danger")
        
    elif(response[0]==1):
        result1=dbc.Alert("It is a positive review", color="success")
        
    else:
        result1='unknown '
        
        
    return result1


# In[ ]:





# In[ ]:





# In[ ]:


# main function to control flow of your project
def main():
    # pass   # we use pass when we dont know what to write in the function
    
    # life cycle of the project starts from here
    print("start of my project")
    load_model()
    open_browser()
    
   
    
    
    global project_name  # to use global variable we shld use use global keyword
    global scrappedReviews
    global app
    global stopwords
    stopwords=set(STOPWORDS)
    global filename
    filename='wordcloud.jpg'
    global count
    count=pd.read_csv('scrapped_reviews_positivity')
    global df
    df=pd.read_csv('scrappedReviews.csv')
    global image_filename
    image_filename = mywordcloud(df,title=None) # replace with your own image
    global encoded_image
    encoded_image = base64.b64encode(open(image_filename, 'rb').read())
    
    
    
    
    project_name="sentiment analysis "
    print("project name is ",project_name)
    
    app.title=project_name
    app.layout=create_app_ui()
    app.run_server()  # blocking statement, all th code after this wont get executed
    
    
    
    print("end of the project")
    # life cycle of the project ends from here
    project_name=None   # after project is ended we shld reinitilized it for goog memory managment
    


# In[ ]:


# calling the main function
if __name__ == '__main__':
    main()




# In[ ]:





# In[ ]:





# In[ ]:




