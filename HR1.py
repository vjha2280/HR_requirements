import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

def model_neural_network(df_train,y_train,X_val,y_val):
    n_features = df_train.shape[1]
    model = Sequential()
    model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
    model.add(Dropout(0.25))
    model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    es = EarlyStopping(monitor='val_loss', patience=20)
    history = model.fit(X_train, y_train,validation_data=(X_val,y_val) , epochs=100, batch_size=64, verbose=0,callbacks=[es])
    return model ,history

def plot_accuracy(history):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def plot_loss(history):
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def model_evaluate(model,X_train,y_train,X_test,y_test):
    train_loss ,train_acc = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
    yhat = model.predict_classes(X_test)
    print(confusion_matrix(y_test, yhat))
    print('Precision Score:',precision_score(y_test, yhat))
    print('Recall Score:',recall_score(y_test, yhat))
    print('F1 score:',f1_score(y_test, yhat))

def data_split(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
    return X_train , y_train , X_test, y_test , X_val ,y_val

if __name__ == "__main__":
    df_clean = pd.read_csv('clean_data.csv')
    df_clean.set_index('enrollee_id',inplace=True)
    df_train = df_clean[df_clean.target.notnull()]
    df_test = df_clean[df_clean.target.isnull()]
    X = df_train.loc[:, df_train.columns != 'target']
    y = df_train.loc[:,'target']
    df_test = df_test.drop('target', axis=1)
    X_train, y_train, X_test, y_test, X_val, y_val = data_split(X,y)
    model , history = model_neural_network(X_train,y_train,X_val,y_val)
    plot_accuracy(history)
    plot_loss(history)
    model_evaluate(model, X_train, y_train, X_test, y_test)