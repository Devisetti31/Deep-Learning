import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import sklearn
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from keras.models import Sequential
from keras.layers import InputLayer, Dense
from keras.optimizers import SGD

from mlxtend.plotting import plot_decision_regions

# Read datasets
df1 = pd.read_csv("concerticcir1.csv",header=None)
df2 = pd.read_csv("concertriccir2.csv",header=None)
df3 = pd.read_csv("linearsep.csv",header=None)
df4 = pd.read_csv("outlier.csv",header=None)
df5 = pd.read_csv("overlap.csv",header=None)
df6 = pd.read_csv("twospirals.csv",header=None)
df7 = pd.read_csv("ushape.csv",header=None)
df8 = pd.read_csv("xor.csv",header=None)

def page_2_function():
    option = st.sidebar.radio("CHOOSE DATASET", ("CONCENTRIC CIR1", "CONCENTRIC CIR2", "LINEAR SEP", "OUTLIER",
                                             "OVERLAP", "TWO SPIRAL", "USHAPE", "XOR"))

    if option == "CONCENTRIC CIR1":
        data = df1
    elif option == "CONCENTRIC CIR2":
        data = df2
    elif option == "LINEAR SEP":
        data = df3
    elif option == "OUTLIER":
        data = df4
    elif option == "OVERLAP":
        data = df5
    elif option == "TWO SPIRAL":
        data = df6
    elif option == "USHAPE":
        data = df7
    elif option == "XOR":
        data = df8

# Sidebar options
    hl1 = st.sidebar.selectbox('Choose number of Hidden layers', (2, 3, 4, 5, 6))
    af1 = st.sidebar.selectbox('Choose activation function', ('sigmoid', 'tanh'))
    rs1 = st.sidebar.checkbox('Are you using Random State?')
    bias1 = st.sidebar.checkbox('Are you using Bias?')
    lr1 = st.sidebar.selectbox('Choose learning rate', (0.00001, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3))
    epoc1 = int(st.sidebar.number_input("Enter number of epochs", 10, 500))
    batch_size1 = int(st.sidebar.number_input("Enter batch size", 10, 300))
    valid_split1 = st.sidebar.number_input("Split validation data", 0.1, 0.5)

# Submission button
    if st.sidebar.button('Submit'):

        txt = f'TensorFlow Playground for {option.lower()} Data Frame'
        styled_text = f""" <div style="font-size: 30px; color: black; text-align: left;">
                                     <span style="font-weight: bold;">{txt}</span> 
                            </div>"""

# Display the styled text using Markdown
        st.markdown(styled_text, unsafe_allow_html=True)

        st.write("### Data Visualization")

        with st.spinner('Please wait ...'):

    # Data preparation
            fv = data.iloc[:, :-1]
            cv = data.iloc[:, -1]
            cv = cv.astype('int')  # Map values to integers

        

        # Create subplots
            fig, axes = plt.subplots(1, 2, figsize=(12, 8))

        # Scatter plot
            sns.scatterplot(x=fv.iloc[:, 0], y=fv.iloc[:, 1], hue=cv, ax=axes[0])
            axes[0].set_title('SCATTER PLOT FOR DATA')

        # Train-test split
            x_train, x_test, y_train, y_test = train_test_split(fv, cv, test_size=0.2, stratify=cv, random_state=rs1)

    #scaling
            std = StandardScaler()
            x_train = std.fit_transform(x_train)
            x_test = std.transform(x_test)

        # Model building
            model = Sequential()
            model.add(InputLayer(input_shape=(2,)))

            neurons = 2 * hl1
            for i in range(hl1):
                model.add(Dense(units=neurons, activation=af1, use_bias=bias1))
                neurons -= 2

            model.add(Dense(units=1, activation='sigmoid', use_bias=bias1))

            model.compile(optimizer=SGD(learning_rate=lr1), loss='binary_crossentropy', metrics=['accuracy'])

    # Model training
            history = model.fit(x_train, y_train, epochs=epoc1, batch_size=batch_size1, validation_split=valid_split1)

            # Plot decision regions for binary classification
            plot_decision_regions(x_test, y_test.values, clf=model, ax=axes[1])
            axes[1].set_title('DECISION REGION FOR TEST DATA')
            st.pyplot(fig)

        # Model evaluation
            st.write('                  ')
            st.write('                  ')
            st.write("### Model Evaluation Plots")

            fig3, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot training and validation accuracy
            losses = history.history['accuracy']
            valid = history.history['val_accuracy']
            axes[0].plot(range(1, epoc1+1), losses, label='Training Accuracy')
            axes[0].plot(range(1, epoc1+1), valid, label='Validation Accuracy')
            axes[0].set_title('Accuracy Plot')
            axes[0].legend()

# Plot training and validation loss
            losses1 = history.history['loss']
            valid1 = history.history['val_loss']
            axes[1].plot(range(1, epoc1+1), losses1, label='Training Loss')
            axes[1].plot(range(1, epoc1+1), valid1, label='Validation Loss')
            axes[1].set_title('Loss Plot')
            axes[1].legend()

        # Show the plots
            st.pyplot(fig3)

            st.write('                  ')
            st.write('                  ')
            st.write("### Model Evaluation Score")

            loss1, accuracy1 = model.evaluate(x_test, y_test)
            loss2, accuracy2 = model.evaluate(x_train, y_train)

            train_accuracy_color = 'green'  
            test_accuracy_color = 'green'   
            train_loss_color = 'red'
            test_loss_color = 'red'

        
            st.markdown("""
                <div style="display: flex; justify-content: space-between;">
                    <div style="text-align: left;">
                        <h2 style="font-size: 25px; color: {};">Train Accuracy: {:.2f}</h2>
                        <h2 style="font-size: 25px; color: {};">Train Loss: {:.2f}</h2>
                    </div>
                    <div style="text-align: right;">
                        <h2 style="font-size: 25px; color: {};">Test Accuracy: {:.2f}</h2>
                        <h2 style="font-size: 25px; color: {};">Test Loss: {:.2f}</h2>
                    </div>
                </div>
                """.format(train_accuracy_color, accuracy2, train_loss_color, loss2, test_accuracy_color, accuracy1, test_loss_color, loss1), unsafe_allow_html=True)
            

if __name__ == "__main__":
    page_2_function()