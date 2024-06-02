import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from proposed_model import ProposedModelFrameWork
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns



def plot_class_wise_metric(classes,y_test,y_pred_classes):
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred_classes, average=None)
    accuracy = accuracy_score(y_test, y_pred_classes)
    class_labels=[]
    
    # Define class labels
    if classes==2:
        class_labels=['Epileptic','Non-Epileptic']
    else:
        class_labels = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5']
    
    # Define metrics to plot
    metrics = ['Precision', 'Recall', 'F1-score', 'Accuracy']
    classwise_metrics = [precision, recall, f1_score, [accuracy]*len(precision)]
    
    # Plot bar graph
    fig, ax = plt.subplots(figsize=(6, 4))
    
    index = np.arange(len(class_labels))
    bar_width = 0.2
    opacity = 0.8
    
    for i, metric in enumerate(metrics):
        ax.bar(index + i * bar_width, classwise_metrics[i], bar_width, label=metric)
    
    ax.set_xlabel('Classes')
    ax.set_ylabel('Scores')
    ax.set_title('Class-wise Metrics')
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(class_labels)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{classes}_class_bargraph.png')
    plt.show()

def plot_loss_curve(history, classes):

    # Access loss and accuracy from training history
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    epochs = range(1, len(loss) + 1)
   
   # Plot loss vs epoch
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{classes}_class_loss curve.png')
    plt.show()

# Plot accuracy vs epoch
    plt.plot(epochs, accuracy, 'b', label='Training Accuracy')
    plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'{classes}_class_accuracy.png')
    plt.show()



def plot_confusion_matrix(y_test, y_pred_classes):
    cm = confusion_matrix(y_test, y_pred_classes)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.set_theme(font_scale=1.2)  # Adjust font scale for better readability
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    
    
    
def metric_values(model,y_pred_classes,X_test,y_test):
    accuracy = model.evaluate(X_test,y_test)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred_classes, average='weighted')
    print("Accuracy:", accuracy*100,"%")
    print("Precision:", precision*100,"%")
    print("Recall:", recall*100,"%")
    print("F1-score:", f1_score*100,"%")


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description='Process a CSV file and perform some operations.')
    
    # Add arguments
    parser.add_argument('-d', '--data', required=True, type=str, 
                        help='The path to the data CSV file')
    parser.add_argument('-p', '--classes', required=True, type=int, 
                        help='The no. of the classes')
    
    
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Use the arguments
    data_file = args.data
    classes=args.classes

    
    # Data Preprocessing
    df = pd.read_csv(f'./dataset/{data_file}')
    df.drop(df.columns[0], axis=1, inplace=True)
    X=df.drop('y',axis=1)
    y=df['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    scaler = StandardScaler()

# Fit the scaler on the data and transform it
    X_train = scaler.fit_transform(X_train)

# Convert the scaled array back to a DataFrame
    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_test=scaler.transform(X_test)
    X_test=pd.DataFrame(X_test, columns=X.columns)

    if classes==2:
        y_train = y_train.mask(y_train > 1, 0)
        y_test= y_test.mask(y_test>1,0)
    else:
        y_train = y_train-1
        y_test= y_test-1
    

    X_train_values = X_train.values
    y_train_values = np.array(y_train)

    # Reshape input data for 1D CNN
    X_cnn = X_train_values.reshape(X_train_values.shape[0], X_train_values.shape[1], 1)
    input_shape=(X_train_values.shape[1], 1)

    model= ProposedModelFrameWork(classes,input_shape)
    epochs=50
    batch_size=32
    validation_split=0.2
    history=model.train(X_cnn, y_train_values, epochs, batch_size, validation_split)
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)  # Convert predicted probabilities to class labels
    metric_values(model,y_pred_classes,X_test,y_test)
    plot_confusion_matrix(y_test,y_pred_classes)
    plot_loss_curve(history,classes)
    plot_class_wise_metric(classes,y_test,y_pred_classes)


    


    
    
    
    

if __name__ == '__main__':
    main()
