from sklearn.metrics import accuracy_score

def predict_and_eval(model, X_test, y_test):
    # Predict using the model
    y_pred = model.predict(X_test)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy
