from methods import *


from spacy.training import offsets_to_biluo_tags


def get_cleaned_label(label: str):
    if "-" in label:
        return label.split("-")[1]
    else:
        return label


def create_total_target_vector(docs, nlp):
    target_vector = []
    for doc in docs:

        new = nlp.make_doc(doc[0])
        entities = doc[1]["entities"]
        bilou_entities = offsets_to_biluo_tags(new, entities)
        final = []
        for item in bilou_entities:
            final.append(get_cleaned_label(item))
        target_vector.extend(final)
    return target_vector


def create_prediction_vector(text, nlp):
    return [get_cleaned_label(prediction) for prediction in get_all_ner_predictions(text, nlp)]


def create_total_prediction_vector(docs: list, nlp):
    prediction_vector = []
    for doc in docs:
        prediction_vector.extend(create_prediction_vector(doc[0], nlp))
    return prediction_vector


def get_all_ner_predictions(text, nlp):
    doc = nlp(text)
    entities = [(e.start_char, e.end_char, e.label_) for e in doc.ents]
    bilou_entities = offsets_to_biluo_tags(doc, entities)
    return bilou_entities


def get_model_labels(nlp):
    labels = list(nlp.get_pipe("ner").labels)
    labels.append("O")
    return sorted(labels)


def get_dataset_labels(docs, nlp):
    return sorted(set(create_total_target_vector(docs, nlp)))


from sklearn.metrics import confusion_matrix,classification_report

def generate_confusion_matrix(docs, nlp):
    classes = sorted(set(create_total_target_vector(docs, nlp)))
    y_true = create_total_target_vector(docs, nlp)
    y_pred = create_total_prediction_vector(docs, nlp)




    return confusion_matrix(y_true, y_pred, labels=classes),classification_report(y_true= y_true,y_pred= y_pred,labels=classes,output_dict=True)


from matplotlib import pyplot
import numpy


def plot_confusion_matrix(docs, classes, nlp, normalize=False, cmap=pyplot.cm.Blues, ):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    title = 'Confusion Matrix'

    # Compute confusion matrix
    cm,report = generate_confusion_matrix(docs, nlp)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]

    fig, ax = pyplot.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=numpy.arange(cm.shape[1]),
           yticks=numpy.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    pyplot.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return report, pyplot


def get_metrics(y_true,y_pred,classes,normalize=False,cmap=pyplot.cm.Blues):
    title = 'Confusion Matrix'



    cm=confusion_matrix(y_true= y_true,y_pred= y_pred, labels=classes)
    report=classification_report(y_true=y_true, y_pred=y_pred,labels=classes, output_dict=True)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]

    fig, ax = pyplot.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=numpy.arange(cm.shape[1]),
           yticks=numpy.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    pyplot.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return report, pyplot


import spacy_stanza


def eval_spacy(docs, model_name):
    nlp = spacy.load(model_name)
    return plot_confusion_matrix(docs, classes=get_dataset_labels(docs, nlp), nlp=nlp, normalize=False, )


def eval_stanza(docs, model_name):
    nlp = spacy_stanza.load_pipeline('en', processors='tokenize,ner',
                                     dir=model_name,
                                     )
    return plot_confusion_matrix(docs, classes=get_dataset_labels(docs, nlp), nlp=nlp, normalize=False, )

