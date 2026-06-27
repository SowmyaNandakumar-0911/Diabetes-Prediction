import sys
from src.diabetesproject.components.data_ingestion import DataIngestion
from src.diabetesproject.components.data_transformation import DataTransformation
from src.diabetesproject.components.model_trainer import ModelTrainer
from src.diabetesproject.exception import CustomException

class TrainPipeline:
    def run_pipeline(self):
        try:
            data_ingestion = DataIngestion()
            train_data, test_data = data_ingestion.initiate_data_ingestion()

            data_transformation = DataTransformation()
            train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

            model_trainer = ModelTrainer()
            accuracy, model_name = model_trainer.initiate_model_trainer(train_arr, test_arr)

            print(f"✅ Training complete! Best Model: {model_name} | Accuracy: {accuracy:.4f}")
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()