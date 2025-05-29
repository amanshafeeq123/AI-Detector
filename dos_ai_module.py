import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from tensorflow import keras
import joblib
import os
from datetime import datetime
import logging

# sets up the logging import
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

class DosAIDetector:
    def __init__(self, output_dir='output'):
        """Initialize the DoS AI Detector with both ML and DL models."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # initializes the scalers and encoders
        self.scalers = {
            'numeric': StandardScaler(),
            'time': StandardScaler()
        }
        self.encoders = {
            'source': LabelEncoder(),
            'destination': LabelEncoder(),
            'protocol': LabelEncoder()
        }
        
        # initializes the models
        self.isolation_forest = IsolationForest(
            n_estimators=100,
            contamination=0.1,
            random_state=42
        )
        
        # will initialize when needed
        self.autoencoder = None
        
        # default training configuration
        self.training_config = {
            'epochs': 30,
            'update_frequency': 5
        }
        
    def load_csv(self, file_path):
        """Load and validate CSV file."""
        try:
            df = pd.read_csv(file_path)
            required_columns = ['Time', 'Source', 'Destination', 'Protocol', 'Length']
            
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
            
            
            if df.empty:
                raise ValueError("CSV file is empty")
            
            logging.info(f"Successfully loaded CSV with {len(df)} rows")
            return df
            
        except Exception as e:
            logging.error(f"Error loading CSV: {str(e)}")
            raise
    
    def preprocess_data(self, df):
        """Preprocess the data for ML/DL models."""
        try:
            # creates a copy to avoid modifying the original
            processed_df = df.copy()
            
            # converts the time to seconds from start
            processed_df['timestamp'] = pd.to_numeric(processed_df['Time'], errors='coerce')
            processed_df['timestamp'] = processed_df['timestamp'] - processed_df['timestamp'].min()
            
            # handles the categorical variables with consistent column names
            categorical_columns = {
                'Source': 'source_encoded',
                'Destination': 'destination_encoded',
                'Protocol': 'protocol_encoded'
            }
            
            for original_col, encoded_col in categorical_columns.items():
                if original_col in processed_df.columns:
                    processed_df[encoded_col] = self.encoders[original_col.lower()].fit_transform(processed_df[original_col])
                else:
                    logging.warning(f"Column {original_col} not found in DataFrame")
                    processed_df[encoded_col] = 0  # default value if column missing
            
            # scales the numeric features
            numeric_features = ['Length', 'timestamp']
            processed_df[numeric_features] = self.scalers['numeric'].fit_transform(processed_df[numeric_features])
            
            # generates the derived features
            processed_df = self._generate_derived_features(processed_df)
            
            logging.info("Data preprocessing completed successfully")
            return processed_df
            
        except Exception as e:
            logging.error(f"Error in preprocessing: {str(e)}")
            raise
    
    def _generate_derived_features(self, df):
        """Generate derived features for anomaly detection."""
        try:
            # calculates the packets per second
            df['time_window'] = df['timestamp'].astype(int)
            packets_per_sec = df.groupby('time_window').size().reset_index(name='packets_per_sec')
            df = df.merge(packets_per_sec, on='time_window', how='left')
            
            # calculates the unique sources per time window
            unique_sources = df.groupby('time_window')['Source'].nunique().reset_index(name='unique_sources')
            df = df.merge(unique_sources, on='time_window', how='left')
            
            # calculates the packet burst rate (rolling window)
            df['burst_rate'] = df.groupby('time_window')['Length'].transform('sum')
            
            # ensures all required columns exist
            required_columns = ['Length', 'packets_per_sec', 'unique_sources', 'burst_rate',
                              'source_encoded', 'destination_encoded', 'protocol_encoded']
            
            for col in required_columns:
                if col not in df.columns:
                    logging.warning(f"Column {col} not found, adding with zeros")
                    df[col] = 0
            
            return df
            
        except Exception as e:
            logging.error(f"Error generating derived features: {str(e)}")
            raise
    
    def create_autoencoder(self, input_dim):
        """Create and compile autoencoder model."""
        encoder = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(8, activation='relu')
        ])
        
        decoder = keras.Sequential([
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(input_dim, activation='sigmoid')
        ])
        
        autoencoder = keras.Sequential([encoder, decoder])
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return autoencoder
    
    def train_models(self, df):
        """Train both IsolationForest and Autoencoder models."""
        try:
            # verifies and prepares the feature matrix
            features = ['Length', 'packets_per_sec', 'unique_sources', 'burst_rate',
                       'source_encoded', 'destination_encoded', 'protocol_encoded']
            
            # checks if all features exist
            missing_features = [f for f in features if f not in df.columns]
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            
            X = df[features].values
            logging.info(f"Prepared feature matrix with shape: {X.shape}")
            
            # trains the IsolationForest with progress updates
            logging.info("Training IsolationForest...")
            n_samples = len(X)
            logging.info(f"Training on {n_samples} samples")
            
            if n_samples > 100000:
                logging.info("Large dataset detected. Using subsample for initial training...")
                subsample_size = min(100000, n_samples // 10)
                indices = np.random.choice(n_samples, subsample_size, replace=False)
                X_subsample = X[indices]
                self.isolation_forest.fit(X_subsample)
                logging.info("Initial training complete, predicting on full dataset...")
                if_predictions = self.isolation_forest.predict(X)
            else:
                if_predictions = self.isolation_forest.fit_predict(X)
            
            if_scores = self.isolation_forest.score_samples(X)
            logging.info("IsolationForest training completed")
            
            # trains the Autoencoder with progress callback
            logging.info("Training Autoencoder...")
            if self.autoencoder is None:
                self.autoencoder = self.create_autoencoder(X.shape[1])
            
            class TrainingCallback(keras.callbacks.Callback):
                def __init__(self, update_frequency, total_epochs):
                    super().__init__()
                    self.update_frequency = update_frequency
                    self.total_epochs = total_epochs
                
                def on_epoch_end(self, epoch, logs=None):
                    if (epoch + 1) % self.update_frequency == 0:
                        logging.info(f"Autoencoder training epoch {epoch + 1}/{self.total_epochs}, loss: {logs['loss']:.4f}")
            
            # determines the batch size based on the dataset size
            batch_size = min(32, max(16, n_samples // 1000))
            logging.info(f"Using batch size of {batch_size}")
            
            history = self.autoencoder.fit(
                X, X,
                epochs=self.training_config['epochs'],
                batch_size=batch_size,
                validation_split=0.1,
                verbose=0,
                callbacks=[TrainingCallback(self.training_config['update_frequency'], 
                                         self.training_config['epochs'])]
            )
            
            logging.info("Autoencoder training completed")
            
            # gets the autoencoder predictions with progress updates
            logging.info("Generating autoencoder predictions...")
            batch_size = min(1000, n_samples)
            n_batches = (n_samples + batch_size - 1) // batch_size
            reconstructed = np.zeros_like(X)
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                batch_X = X[start_idx:end_idx]
                reconstructed[start_idx:end_idx] = self.autoencoder.predict(batch_X, verbose=0)
                if (i + 1) % 10 == 0:
                    logging.info(f"Processed {end_idx}/{n_samples} samples")
            
            mse = np.mean(np.power(X - reconstructed, 2), axis=1)
            threshold = np.percentile(mse, 90)  # Top 10% as anomalies
            ae_predictions = (mse > threshold).astype(int)
            
            # combines the predictions
            df['anomaly_score_if'] = if_scores
            df['anomaly_score_ae'] = mse
            df['is_anomaly_if'] = if_predictions == -1
            df['is_anomaly_ae'] = ae_predictions == 1
            df['is_anomaly'] = df['is_anomaly_if'] | df['is_anomaly_ae']
            
            # calculates accuracy metrics
            self.metrics = {
                'isolation_forest_mse': np.mean(np.power(if_scores - np.mean(if_scores), 2)),
                'autoencoder_mse': np.mean(mse),
                'isolation_forest_accuracy': np.mean(if_predictions == -1) * 100,
                'autoencoder_accuracy': np.mean(ae_predictions) * 100,
                'combined_accuracy': np.mean(df['is_anomaly']) * 100
            }
            
            # logs the metrics
            logging.info("\nModel Performance Metrics:")
            logging.info(f"Isolation Forest MSE: {self.metrics['isolation_forest_mse']:.4f}")
            logging.info(f"Autoencoder MSE: {self.metrics['autoencoder_mse']:.4f}")
            logging.info(f"Isolation Forest Accuracy: {self.metrics['isolation_forest_accuracy']:.2f}%")
            logging.info(f"Autoencoder Accuracy: {self.metrics['autoencoder_accuracy']:.2f}%")
            logging.info(f"Combined Model Accuracy: {self.metrics['combined_accuracy']:.2f}%")
            
            # logs the summary statistics
            n_anomalies = df['is_anomaly'].sum()
            anomaly_percentage = (n_anomalies / len(df)) * 100
            logging.info(f"Analysis complete: {n_anomalies} anomalies detected ({anomaly_percentage:.2f}%)")
            
            return df, history
            
        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")
            raise
    
    def generate_visualizations(self, df):
        """Generate and save visualization plots."""
        try:
            plots = {}
            
            # 1. packet length/count over time
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df['timestamp'], df['Length'], label='Packet Length', alpha=0.6)
            anomalies = df[df['is_anomaly']]
            ax.scatter(anomalies['timestamp'], anomalies['Length'],
                      color='red', label='Anomalies', alpha=0.5)
            ax.set_title('Packet Length Over Time with Anomalies')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Packet Length')
            ax.legend()
            plots['timeline'] = fig
            
            # 2. top source ips during anomalies
            fig, ax = plt.subplots(figsize=(10, 6))
            top_sources = df[df['is_anomaly']]['Source'].value_counts().head(10)
            sns.barplot(x=top_sources.values, y=top_sources.index, ax=ax)
            ax.set_title('Top Source IPs During Anomalies')
            ax.set_xlabel('Frequency')
            plots['top_sources'] = fig
            
            # 3. feature correlation heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            correlation_features = ['Length', 'packets_per_sec', 'unique_sources',
                                  'burst_rate', 'anomaly_score_if', 'anomaly_score_ae']
            corr_matrix = df[correlation_features].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
            ax.set_title('Feature Correlation Heatmap')
            plots['correlation'] = fig
            
            # 4. anomaly score distribution
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            sns.histplot(data=df, x='anomaly_score_if', ax=ax1)
            ax1.set_title('IsolationForest Anomaly Score Distribution')
            sns.histplot(data=df, x='anomaly_score_ae', ax=ax2)
            ax2.set_title('Autoencoder Reconstruction Error Distribution')
            plots['scores'] = fig
            
            # saves the plots
            for name, fig in plots.items():
                fig.savefig(os.path.join(self.output_dir, f'{name}.png'))
                plt.close(fig)
            
            logging.info("Visualizations generated and saved successfully")
            return plots
            
        except Exception as e:
            logging.error(f"Error generating visualizations: {str(e)}")
            raise
    
    def save_models(self):
        """Save trained models for later use."""
        try:
            # saves the IsolationForest
            joblib.dump(self.isolation_forest,
                       os.path.join(self.output_dir, 'isolation_forest.joblib'))
            
            # saves the Autoencoder
            if self.autoencoder:
                self.autoencoder.save(os.path.join(self.output_dir, 'autoencoder.keras'))
            
            # saves the encoders and scalers
            joblib.dump(self.encoders,
                       os.path.join(self.output_dir, 'encoders.joblib'))
            joblib.dump(self.scalers,
                       os.path.join(self.output_dir, 'scalers.joblib'))
            
            logging.info("Models saved successfully")
            
        except Exception as e:
            logging.error(f"Error saving models: {str(e)}")
            raise
    
    def summarize_results(self, df):
        """Generate a summary of the analysis results."""
        try:
            # calculates basic statistics
            total_packets = len(df)
            anomalies = df['is_anomaly'].sum()
            anomaly_percentage = (anomalies / total_packets) * 100
            
            # gets protocol distribution
            protocol_dist = df['Protocol'].value_counts().to_dict()
            
            # creates the summary dictionary
            summary = {
                'total_packets': total_packets,
                'anomalies_detected': int(anomalies),
                'anomaly_percentage': f"{anomaly_percentage:.2f}%",
                'protocols': protocol_dist,
                'model_metrics': {
                    'isolation_forest_mse': f"{self.metrics['isolation_forest_mse']:.4f}",
                    'autoencoder_mse': f"{self.metrics['autoencoder_mse']:.4f}",
                    'isolation_forest_accuracy': f"{self.metrics['isolation_forest_accuracy']:.2f}%",
                    'autoencoder_accuracy': f"{self.metrics['autoencoder_accuracy']:.2f}%",
                    'combined_accuracy': f"{self.metrics['combined_accuracy']:.2f}%"
                }
            }
            
            return summary
            
        except Exception as e:
            logging.error(f"Error generating summary: {str(e)}")
            raise

def main():
    """Main function to demonstrate usage."""
    try:
        # initializes the detector
        detector = DosAIDetector()
        
        # gets the input file
        file_path = input("Enter the path to your Wireshark CSV export file: ")
        
        # processes the data
        df = detector.load_csv(file_path)
        processed_df = detector.preprocess_data(df)
        
        # trains the models and gets the predictions
        results_df, history = detector.train_models(processed_df)
        
        # generates the visualizations
        plots = detector.generate_visualizations(results_df)
        
        # saves the models
        detector.save_models()
        
        # gets the summary
        summary = detector.summarize_results(results_df)
        
        print("\nAnalysis completed successfully!")
        print(f"Results saved in: {detector.output_dir}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        logging.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main() 
