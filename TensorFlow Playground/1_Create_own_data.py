import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

import sklearn
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from keras.models import Sequential
from keras.layers import InputLayer, Dense
from keras.optimizers import SGD

from mlxtend.plotting import plot_decision_regions

def page_1_function():

    sam = st.sidebar.number_input("Enter number of samples", 100, 10000)
    cla = st.sidebar.number_input('Enter number of classes', 2, 2)

    ts = st.sidebar.number_input("Split x_test_size", 0.2, 0.5)
    rs = st.sidebar.checkbox('Are you using Random State?')

    if rs:
        rs = True
    else:
        rs = False

    hl1 = st.sidebar.selectbox('Choose number of Hidden layers', (2, 3, 4, 5, 6))

    af1 = st.sidebar.selectbox('Choose activation function', ('sigmoid', 'tanh'))

    bias1 = st.sidebar.checkbox('Are you using Bias?')

    if bias1:
        bias1 = True
    else:
        bias1 = False

    lr1 = st.sidebar.selectbox('Choose learning rate', (0.00001, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3))

    epoc1 = int(st.sidebar.number_input("Enter number of epochs", 10, 500))

    batch_size1 = int(st.sidebar.number_input("Enter batch size", 10, 300))

    valid_split1 = st.sidebar.number_input("Split validation data", 0.1, 0.5)

    if st.sidebar.button("SUBMIT"):
        txt = f'TensorFlow Playground for Our Own Data Frame'
        styled_text = f""" <div style="font-size: 30px; color: black; text-align: left;">
                                     <span style="font-weight: bold;">{txt}</span> 
                            </div>"""

        st.markdown(styled_text, unsafe_allow_html=True)

        st.write("### Data Visualization")

        with st.spinner('Please wait ...'):
        
            fv, cv = make_classification(n_samples=sam, n_features=2, n_informative=2, n_classes=cla, n_redundant=0,
                                      n_clusters_per_class=1, n_repeated=0, class_sep=0.9)
            

        # Create subplots
            fig, axes = plt.subplots(1, 2, figsize=(12, 8))

        # Scatter plot
            sns.scatterplot(x=fv[:, 0], y=fv[:, 1], hue=cv, ax=axes[0])
            axes[0].set_title('SCATTER PLOT FOR DATA')

        # Train-test split
            x_train, x_test, y_train, y_test = train_test_split(fv, cv, test_size=ts, stratify=cv, random_state=rs)

        # Scaling
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
            plot_decision_regions(x_test, y_test, clf=model, ax=axes[1])
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
    page_1_function()
