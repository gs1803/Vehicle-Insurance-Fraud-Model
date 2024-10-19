import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package='Custom', name='FocalLoss')
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2., alpha=0.25, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, name="focal_loss"):
        super(FocalLoss, self).__init__(reduction=reduction, name=name)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        bce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        bce_exp = tf.exp(-bce)
        focal_loss = self.alpha * (1 - bce_exp) ** self.gamma * bce

        return focal_loss
    

@tf.keras.utils.register_keras_serializable(package="Custom", name="f1_score")
def f1_score(y_true, y_pred):
    f1_scores = []

    for i in range(2):
        y_true_binary = tf.cast(tf.equal(y_true, i), tf.int32)
        y_pred_binary = tf.cast(tf.equal(tf.argmax(y_pred, axis=-1), i), tf.int32)

        true_positives = tf.reduce_sum(tf.cast(y_true_binary * y_pred_binary, tf.float32))
        predicted_positives = tf.reduce_sum(tf.cast(y_pred_binary, tf.float32))
        possible_positives = tf.reduce_sum(tf.cast(y_true_binary, tf.float32))

        precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
        recall = true_positives / (possible_positives + tf.keras.backend.epsilon())

        f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
        
        f1_scores.append(f1)

    macro_f1 = tf.reduce_mean(f1_scores)
    
    return macro_f1