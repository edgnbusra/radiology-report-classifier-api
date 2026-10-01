from transformers import pipeline

classifier = pipeline("text-classification")

print(classifier("The scan shows no abnormal findings."))

