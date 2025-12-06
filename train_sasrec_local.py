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
import pickle
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import tensorflow as tf

# Configure TensorFlow to use single thread (to avoid mutex issues)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# Set TensorFlow log level
tf.get_logger().setLevel("INFO")

# Need to set PYTHONPATH first to import these modules
try:
    from sasrec_model import CustomSASRec
    from sasrec_data import CustomSASRecDataSet
    from recommenders.models.sasrec.sampler import WarpSampler
except ImportError as e:
    print(f"Import error: {e}")
    print("Please set PYTHONPATH first:")
    print("export PYTHONPATH=/Users/xianfeng_pei/Desktop/ZDF/recommendations-models-sasrec/docker/code:$PYTHONPATH")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
    sampler = WarpSampler(
        user_data,
        num_users,
        num_items,
        batch_size=hyperparams['batch_size'],
        maxlen=hyperparams['maxlen'],
        n_workers=2,
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

