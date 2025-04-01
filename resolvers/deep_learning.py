"""
Deep Learning-based entity resolution approach.
"""

import logging
import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple, Set, Any, Optional

from resolvers.ml_based import MLEntityResolver

logger = logging.getLogger("entity_resolution")

# Check if TensorFlow is available
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import Dense, Dropout, Input, concatenate, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False
    logger.warning("TensorFlow not available. DeepLearningResolver will fall back to ML approach.")

class DeepLearningResolver(MLEntityResolver):
    """Deep Learning-based entity resolution using neural networks for matching."""
    
    def __init__(self, match_threshold=0.5, training_pairs=15000, model_type='siamese'):
        """
        Initialize the deep learning resolver.
        
        Args:
            match_threshold: Threshold for match prediction (default: 0.5)
            training_pairs: Number of pairs to use for training (default: 15000)
            model_type: Type of model architecture to use ('siamese' or 'feedforward') (default: 'siamese')
        """
        super().__init__(match_threshold=match_threshold, training_pairs=training_pairs)
        
        self.name = "Deep Learning Entity Resolution"
        self.description = "Uses neural networks to learn similarity patterns for entity matching"
        self.model_type = model_type
        self.deep_learning_model = None
        self.feature_encoder = None
        self.feature_scaler = None
        
    def _train_similarity_model(self, df: pd.DataFrame, num_pairs=15000) -> None:
        """
        Train a neural network model to predict record matching.
        
        Args:
            df: Preprocessed dataframe
            num_pairs: Number of pairs to use for training
        """
        logger.info(f"[{self.name}] Training neural network similarity model...")
        
        if not DEEP_LEARNING_AVAILABLE:
            logger.warning(f"[{self.name}] TensorFlow not available. Falling back to Random Forest approach.")
            return super()._train_similarity_model(df, num_pairs)
        
        try:    
            # Generate training data
            pairs_df = self._generate_training_pairs(df, num_pairs)
            
            # Prepare features and labels
            feature_cols = [col for col in pairs_df.columns if col not in ['id1', 'id2', 'label']]
            X = pairs_df[feature_cols].values
            y = pairs_df['label'].values
            
            # Scale features for better neural network performance
            from sklearn.preprocessing import StandardScaler
            self.feature_scaler = StandardScaler()
            X_scaled = self.feature_scaler.fit_transform(X)
            
            # Split into training and validation sets
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Create and train the model
            if self.model_type == 'siamese':
                self.deep_learning_model = self._create_siamese_network(len(feature_cols))
            else:
                self.deep_learning_model = self._create_feedforward_network(len(feature_cols))
                
            # Set up early stopping
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=5, restore_best_weights=True
            )
            
            # Train the model
            history = self.deep_learning_model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=30,
                batch_size=64,
                callbacks=[early_stopping],
                class_weight={0: 1, 1: 2}  # Weight positive examples higher
            )
            
            # Evaluate model
            val_loss, val_accuracy = self.deep_learning_model.evaluate(X_val, y_val)
            logger.info(f"[{self.name}] Validation accuracy: {val_accuracy:.4f}")
            
            # Store feature importances (not directly available from neural networks)
            self.feature_importances = dict(zip(feature_cols, [1.0] * len(feature_cols)))
            
        except Exception as e:
            logger.error(f"[{self.name}] Error training deep learning model: {e}")
            logger.warning(f"[{self.name}] Falling back to Random Forest approach.")
            return super()._train_similarity_model(df, num_pairs)
        

    def _create_siamese_network(self, input_dim):
        """
        Create a siamese neural network for learning entity similarities.
        
        Args:
            input_dim: Dimension of input features
            
        Returns:
            Compiled Keras model
        """
        # Define the base encoder network
        encoder = Sequential([
            Dense(64, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            BatchNormalization()
        ])
        
        # Create twin inputs and encode them
        input_a = Input(shape=(input_dim,))
        input_b = Input(shape=(input_dim,))
        
        encoded_a = encoder(input_a)
        encoded_b = encoder(input_b)
        
        # Combine the encoded representations
        merged = concatenate([encoded_a, encoded_b])
        
        # Add classification layers
        x = Dense(32, activation='relu')(merged)
        x = Dropout(0.3)(x)
        output = Dense(1, activation='sigmoid')(x)
        
        # Create and compile the model
        model = Model(inputs=[input_a, input_b], outputs=output)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _create_feedforward_network(self, input_dim):
        """
        Create a simple feedforward neural network for classification.
        
        Args:
            input_dim: Dimension of input features
            
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _predict_matches(self, df: pd.DataFrame, candidate_pairs: Set[Tuple[int, int]], threshold=0.5) -> List[Tuple[int, int, float]]:
        """
        Predict matches among candidate pairs using neural network.
        
        Args:
            df: Preprocessed dataframe
            candidate_pairs: Set of candidate pairs to evaluate
            threshold: Threshold for match prediction
            
        Returns:
            List of tuples (idx1, idx2, score) representing matched pairs
        """
        logger.info(f"[{self.name}] Predicting matches with threshold {threshold}...")
        
        if not DEEP_LEARNING_AVAILABLE or self.deep_learning_model is None:
            # Fall back to ML approach if deep learning isn't available
            return super()._predict_matches(df, candidate_pairs, threshold)
        
        matches = []
        
        # Batch processing to avoid memory issues
        batch_size = 10000
        pairs_list = list(candidate_pairs)
        
        for i in range(0, len(pairs_list), batch_size):
            batch_pairs = pairs_list[i:i+batch_size]
            batch_features = []
            
            # Calculate features for each pair
            for idx1, idx2 in batch_pairs:
                rec1 = self._extract_record_data(df.iloc[idx1])
                rec2 = self._extract_record_data(df.iloc[idx2])
                similarities = self._calculate_similarities(rec1, rec2)
                batch_features.append(similarities)
            
            # Convert to DataFrame
            batch_df = pd.DataFrame(batch_features)
            
            try:
                # Prepare features
                feature_cols = list(self.feature_importances.keys())
                
                # Handle missing columns
                for col in feature_cols:
                    if col not in batch_df.columns:
                        batch_df[col] = 0.0
                
                X = batch_df[feature_cols].values
                
                # Scale features
                X_scaled = self.feature_scaler.transform(X)
                
                # Predict
                proba = self.deep_learning_model.predict(X_scaled, batch_size=256, verbose=0)
                
                # Add matches above threshold
                for j, (idx1, idx2) in enumerate(batch_pairs):
                    if proba[j][0] >= threshold:
                        matches.append((idx1, idx2, float(proba[j][0])))
                        
            except Exception as e:
                logger.error(f"[{self.name}] Error in deep learning prediction: {e}, falling back to ML approach")
                # Fall back to ML approach for this batch
                batch_matches = super()._predict_matches(df, set(batch_pairs), threshold)
                matches.extend(batch_matches)
        
        logger.info(f"[{self.name}] Found {len(matches)} matches using deep learning")
        return matches