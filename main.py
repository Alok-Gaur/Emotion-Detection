import argparse
from ai_model.training.train import train_model
from ai_model.training.test import test_model
from ai_model.prediction.prediction import make_prediction

def main():
    parser = argparse.ArgumentParser(description='Emotion Detection Project')
    parser.add_argument("--mode", type=str, required=True,
                        choices=['train', 'test', 'predict'],
                        help="Mode to run: train, test or predict")
    parser.add_argument("--image_path", type=str, default=None,
                        help="Path to an image for prediction (predict mode only)")
    parser.add_argument("--directory_path", type=str, default="ai_model/data/test",
                        help="Directory of Images (Used only in test and predic mode)")
    parser.add_argument("--train_directory_path", type=str, default="ai_model/data/train",
                        help="Path to the directory to be trained on.(only for train mode)")
    parser.add_argument("--val_directory_path", type=str, default="ai_model/data/test",
                    help="Directory of Images used for validation (Used only in train mode)")
    parser.add_argument("--model_path", type=str, default=None,
                    help="Path of model to be used (Used only in test and predic mode)")
    
    args = parser.parse_args()

    if args.mode == 'train':
        print("Training model....")
        train_model(train_directory_path=args.train_directory_path, val_directory_path=args.val_directory_path)
        print("Training Complete!")
    elif args.mode == 'test':
        if (args.image_path or args.directory_path):
            if args.model_path:
                print("Testing mode....")
                result = test_model(args.image_path, args.directory_path, args.model_path)
                print("Test Result: ", result)
            else:
                print("Please enter model path.")
        else:
            print("Please enter image or directory path!")
    elif args.model == 'predict':
        if (args.image_path or args.directory_path):
            if args.model_path:
                print("Prediction mode....")
                predictions = make_prediction(args.image_path, args.directory_path, args.model_path)
                print("Predictions: ", predictions)
            else:
                print("Please enter model path.")
        else:
            print("Please enter image or directory path!")
    else:
        print("Enter valid mode!")

if __name__ == "__main__":
    main()
    