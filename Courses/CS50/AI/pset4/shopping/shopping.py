import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    with open(filename) as file:
        reader = csv.reader(file)
        next(reader)
        month_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        evidence = []
        labels = []

        for row in reader:
            new_evidence = []
            # Grab initial numbers
            new_evidence.append(int(row[0]))
            new_evidence.append(float(row[1]))
            new_evidence.append(int(row[2]))
            new_evidence.append(float(row[3]))
            new_evidence.append(int(row[4]))
            new_evidence += [float(cell) for cell in row[5:10]]
            # Grab month and convert to int
            new_evidence.append(month_name.index(row[10]))
            # Grab rest of numbers
            new_evidence += [int(cell) for cell in row[11:15]]
            # Grab Visitor type
            new_evidence.append(1 if row[15] == 'Returning_Visitor' else 0)
            # Grab weekend
            new_evidence.append(1 if row[16] == 'TRUE' else 0)

            # Grab label
            new_label = 1 if row[-1] == 'TRUE' else 0

            # Add to existing data
            evidence.append(new_evidence)
            labels.append(new_label)

        data = (evidence, labels)

    return data


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)

    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    tot_true = tot_false = pred_true = pred_false = 0

    for label, prediction in zip(labels, predictions):
        # If true
        if label == 1:
            tot_true += 1
            if prediction == 1:
                pred_true += 1
        # If false
        else:
            tot_false += 1
            if prediction == 0:
                pred_false += 1

    sensitivity = pred_true / tot_true
    specificity = pred_false / tot_false

    return sensitivity, specificity


if __name__ == "__main__":
    main()
