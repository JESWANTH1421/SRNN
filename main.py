from train import train_model
from evaluate import evaluate_model

def main():
    model = train_model()
    evaluate_model(model)

if __name__ == "__main__":
    main()