#!/usr/bin/env python
"""
Local training SASRec model simplified script (based on MovieLens data)

使用方法：
1. Prepare data: Ensure rating_df (training set) and rating_test_df (test set) are available
2. Set environment variables:
   export PYTHONPATH=/Users/xianfeng_pei/Desktop/ZDF/recommendations-models-sasrec/docker/code:$PYTHONPATH
   export SITE=test
3. Run:
   python train_sasrec_local.py --data_dir /path/to/movielens/data --output_dir /path/to/save/model
"""
import argparse
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Dict

# Set environment variables BEFORE importing TensorFlow to avoid mutex issues
# These must be set before ANY TensorFlow-related imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress WARNING messages too
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations that can cause mutex issues
os.environ['TF_DISABLE_MKL'] = '1'  # Disable MKL
os.environ['OMP_NUM_THREADS'] = '1'  # Limit OpenMP threads (critical for multiprocessing)
os.environ['MKL_NUM_THREADS'] = '1'  # Limit MKL threads
os.environ['NUMEXPR_NUM_THREADS'] = '1'  # Limit NumExpr threads
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # Limit OpenBLAS threads
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'  # Limit macOS Accelerate framework threads
os.environ['NUMBA_NUM_THREADS'] = '1'  # Limit Numba threads

# Disable multiprocessing completely to avoid any subprocess issues
# Set multiprocessing start method to 'spawn' to avoid fork-related issues
# This must be done before importing multiprocessing
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    # Already set, ignore
    pass

# Prevent multiprocessing from being used
multiprocessing.freeze_support()

import numpy as np
import pandas as pd

# Initialize logging BEFORE TensorFlow to avoid log-related mutex issues
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Now import TensorFlow with all protections in place
import tensorflow as tf

# Configure TensorFlow to use single thread (to avoid mutex issues)
# This must be done IMMEDIATELY after importing TensorFlow
try:
    # Disable all GPUs first
    tf.config.set_visible_devices([], 'GPU')
except:
    pass

# Set thread configuration
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# Disable TensorFlow optimizations that might use multiple threads
tf.config.optimizer.set_jit(False)  # Disable XLA JIT compilation

# Set TensorFlow log level
tf.get_logger().setLevel("ERROR")  # Only show errors

logging.info("TensorFlow configured: single thread, GPU disabled")

# Need to set PYTHONPATH first to import these modules
# Import with additional protection to avoid mutex issues during import
logging.info("Importing SASRec modules...")
try:
    # Suppress any TensorFlow warnings during import
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from sasrec_model import CustomSASRec
        from sasrec_data import CustomSASRecDataSet
    logging.info("SASRec modules imported successfully")
except ImportError as e:
    print(f"Import error: {e}")
    print("Please set PYTHONPATH first:")
    print("export PYTHONPATH=/Users/xianfeng_pei/Desktop/ZDF/recommendations-models-sasrec/docker/code:$PYTHONPATH")
    sys.exit(1)
except Exception as e:
    logging.error(f"Unexpected error during import: {e}")
    print(f"Error: {e}")
    print("This might be a mutex issue. Try running with:")
    print("export OMP_NUM_THREADS=1")
    print("export MKL_NUM_THREADS=1")
    sys.exit(1)


class SimpleSampler:
    """
    Single-process sampler to avoid multiprocessing mutex issues with TensorFlow.
    This is a drop-in replacement for WarpSampler that doesn't use multiprocessing.
    """
    def __init__(self, user_train, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.user_train = user_train
        self.usernum = usernum
        self.itemnum = itemnum
        self.batch_size = batch_size
        self.maxlen = maxlen
        np.random.seed(np.random.randint(2e9))
        logging.info("Using SimpleSampler (single-process, no multiprocessing)")
    
    def random_neq(self, left, right, s):
        """Sample a random number not in set s"""
        t = np.random.randint(left, right)
        while t in s:
            t = np.random.randint(left, right)
        return t
    
    def sample(self):
        """Sample a single training example"""
        user = np.random.randint(1, self.usernum + 1)
        while len(self.user_train[user]) <= 1:
            user = np.random.randint(1, self.usernum + 1)
        
        seq = np.zeros([self.maxlen], dtype=np.int32)
        pos = np.zeros([self.maxlen], dtype=np.int32)
        neg = np.zeros([self.maxlen], dtype=np.int32)
        nxt = self.user_train[user][-1]
        idx = self.maxlen - 1
        
        ts = set(self.user_train[user])
        for i in reversed(self.user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0:
                neg[idx] = self.random_neq(1, self.itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1:
                break
        
        return (user, seq, pos, neg)
    
    def next_batch(self):
        """Generate next batch of samples"""
        one_batch = []
        for i in range(self.batch_size):
            one_batch.append(self.sample())
        return zip(*one_batch)
    
    def close(self):
        """No-op for compatibility with WarpSampler interface"""
        pass


def prepare_movielens_data(rating_df: pd.DataFrame) -> tuple:
    """
    Convert MovieLens data format to the format required by SASRec
    
    :param rating_df: DataFrame with columns: user_id, movie_id, rating, timestamp
    :return: (data_dict, item_id_for_extid, extid_for_itemid)
    """
    # Filter low ratings (only keep >= 4 as positive feedback)
    filtered = rating_df[rating_df['rating'] >= 4].copy()
    
    # Build user -> [item1, item2, ...] dictionary (sorted by timestamp)
    user_data = {}
    for user_id, group in filtered.sort_values('timestamp').groupby('user_id'):
        user_data[int(user_id)] = group['movie_id'].astype(int).tolist()
    
    # Build item_id mapping (SASRec requires consecutive IDs starting from 1)
    all_items = set()
    for items in user_data.values():
        all_items.update(items)
    
    all_items = sorted(all_items)
    item_id_for_extid = {ext_id: idx + 1 for idx, ext_id in enumerate(all_items)}
    extid_for_itemid = {idx + 1: ext_id for idx, ext_id in enumerate(all_items)}
    
    # Convert external_id in user_data to internal item_id
    user_data_internal = {}
    for user_id, items in user_data.items():
        user_data_internal[user_id] = [
            item_id_for_extid[item] for item in items if item in item_id_for_extid
        ]
    
    num_users = max(user_data_internal.keys()) if user_data_internal else 0
    num_items = len(all_items)
    
    logging.info(f"Data statistics: {num_users} users, {num_items} items")
    
    return user_data_internal, item_id_for_extid, extid_for_itemid, num_users, num_items


def train_sasrec(
    user_data: Dict[int, list],
    num_users: int,
    num_items: int,
    output_dir: str,
    item_id_for_extid: Dict,
    extid_for_itemid: Dict,
    hyperparams: Dict,
):
    """
    Train SASRec model
    
    :param user_data: {user_id: [item_id, ...]} dictionary
    :param num_users: Total number of users
    :param num_items: Total number of items
    :param output_dir: Model save directory
    :param item_id_for_extid: external_id -> internal_id mapping
    :param extid_for_itemid: internal_id -> external_id mapping
    :param hyperparams: Hyperparameters dictionary
    """
    # Create dataset
    dataset = CustomSASRecDataSet(pd.DataFrame({
        'user_id': list(user_data.keys()),
        'item_id': [user_data[uid] for uid in user_data.keys()]
    }))
    dataset.user_data = user_data
    dataset.usernum = num_users
    dataset.itemnum = num_items
    
    # Create sampler
    # Use SimpleSampler instead of WarpSampler to avoid multiprocessing mutex issues
    sampler = SimpleSampler(
        user_data,
        num_users,
        num_items,
        batch_size=hyperparams['batch_size'],
        maxlen=hyperparams['maxlen'],
        n_workers=1,  # Not used, but kept for compatibility
    )
    
    # Create model
    model = CustomSASRec(
        item_num=num_items,
        seq_max_len=hyperparams['maxlen'],
        embedding_dim=hyperparams['hidden_units'],
        attention_dim=hyperparams['hidden_units'],
        conv_dims=[hyperparams['hidden_units'], hyperparams['hidden_units']],
        num_blocks=hyperparams.get('num_blocks', 2),
        num_heads=hyperparams.get('num_heads', 1),
        dropout_rate=hyperparams.get('dropout_rate', 0.2),
        l2_reg=hyperparams.get('l2_emb', 0.0001),
    )
    
    # Training loop
    num_steps = int(len(user_data) / hyperparams['batch_size'])
    optimizer = tf.keras.optimizers.Adam(learning_rate=hyperparams['learning_rate'])
    
    logging.info(f"Training started, {hyperparams['num_epochs']} epochs, {num_steps} steps per epoch")
    
    for epoch in range(1, hyperparams['num_epochs'] + 1):
        epoch_loss = []
        for step in range(num_steps):
            u, seq, pos, neg = sampler.next_batch()
            
            with tf.GradientTape() as tape:
                pos_logits, neg_logits, loss_mask = model({
                    'users': u,
                    'input_seq': seq,
                    'positive': pos,
                    'negative': neg,
                }, training=True)
                
                # Simple BCE loss
                pos_logits = pos_logits[:, 0]
                neg_logits = neg_logits[:, 0]
                loss = -tf.reduce_sum(
                    tf.math.log(tf.math.sigmoid(pos_logits) + 1e-8) * loss_mask
                    - tf.math.log(1 - tf.math.sigmoid(neg_logits) + 1e-8) * loss_mask
                ) / tf.reduce_sum(loss_mask)
                
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            epoch_loss.append(float(loss))
            
            if step % 100 == 0:
                logging.info(f"Epoch {epoch}, Step {step}/{num_steps}, Loss: {loss:.4f}")
        
        avg_loss = np.mean(epoch_loss)
        logging.info(f"Epoch {epoch} completed, average Loss: {avg_loss:.4f}")
    
    # Close sampler to clean up multiprocessing resources
    try:
        sampler.close()
        logging.info("Sampler closed successfully")
    except Exception as e:
        logging.warning(f"Error closing sampler: {e}")
    
    # Save model
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    weights_dir = os.path.join(output_dir, "sasrec_weights")
    model.save_weights(weights_dir)
    logging.info(f"Model weights saved to: {weights_dir}")
    
    # Save metadata
    meta = {
        "item_id_for_extid": item_id_for_extid,
        "external_id_for_item_id": extid_for_itemid,
        "num_items_training": num_items,
        "Sasrec_model_descritption": {
            "maxlen": hyperparams['maxlen'],
            "hidden_units": hyperparams['hidden_units'],
        },
    }
    
    meta_path = os.path.join(output_dir, "meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    logging.info(f"Metadata saved to: {meta_path}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train SASRec model on MovieLens data")
    parser.add_argument("--data_dir", type=str, required=True, help="MovieLens data directory (contains u1.base)")
    parser.add_argument("--output_dir", type=str, required=True, help="Model save directory")
    parser.add_argument("--maxlen", type=int, default=50, help="Sequence maximum length")
    parser.add_argument("--hidden_units", type=int, default=50, help="Hidden units")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=10, help="Training epochs")
    
    args = parser.parse_args()
    
    # Load data
    rating_file = os.path.join(args.data_dir, "u1.base")
    if not os.path.exists(rating_file):
        logging.error(f"Training data file not found: {rating_file}")
        sys.exit(1)
    
    rating_df = pd.read_csv(
        rating_file,
        sep='\t',
        names=['user_id', 'movie_id', 'rating', 'timestamp'],
        encoding='latin-1'
    )
    logging.info(f"Loaded {len(rating_df)} training data")
    
    # Prepare data
    user_data, item_id_for_extid, extid_for_itemid, num_users, num_items = prepare_movielens_data(rating_df)
    
    # Hyperparameters
    hyperparams = {
        'maxlen': args.maxlen,
        'hidden_units': args.hidden_units,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        'num_blocks': 2,
        'num_heads': 1,
        'dropout_rate': 0.2,
        'l2_emb': 0.0001,
    }
    
    # Train
    model = train_sasrec(
        user_data=user_data,
        num_users=num_users,
        num_items=num_items,
        output_dir=args.output_dir,
        item_id_for_extid=item_id_for_extid,
        extid_for_itemid=extid_for_itemid,
        hyperparams=hyperparams,
    )
    
    logging.info(f"Training completed! Model saved to: {args.output_dir}")
    logging.info(f"When using in Lusifer, set: export SASREC_MODEL_DIR={args.output_dir}")


if __name__ == "__main__":
    main()

