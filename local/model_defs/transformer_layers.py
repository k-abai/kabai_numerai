import tensorflow as tf
from tensorflow.keras import layers, models

class FeatureEmbedding(layers.Layer):
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
    def build(self, input_shape):
        self.projection = layers.Dense(self.d_model)
    def call(self, x):
        x = tf.expand_dims(x, axis=-1)
        return self.projection(x)
    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model})
        return config

class LinformerAttention(layers.Layer):
    def __init__(self, d_model, num_heads, k, **kwargs):
        super().__init__(**kwargs)
        self.d_model, self.num_heads, self.k = d_model, num_heads, k
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
    
    def build(self, input_shape):
        seq_len = input_shape[1]
        self.E = self.add_weight(shape=(seq_len, self.k), initializer='glorot_uniform', trainable=True, name='E')
        self.F = self.add_weight(shape=(seq_len, self.k), initializer='glorot_uniform', trainable=True, name='F')

    def call(self, x, training=False):
        k_proj = tf.einsum('bsd,sk->bkd', x, self.E)
        v_proj = tf.einsum('bsd,sk->bkd', x, self.F)
        return self.mha(x, v_proj, k_proj, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "num_heads": self.num_heads, "k": self.k})
        return config

class TransformerEncoderBlock(layers.Layer):
    def __init__(self, d_model, num_heads, ffn_dim, dropout_rate, k=None, **kwargs):
        super().__init__(**kwargs)
        self.d_model, self.num_heads, self.ffn_dim, self.dropout_rate, self.k = d_model, num_heads, ffn_dim, dropout_rate, k
        if k:
            self.attn = LinformerAttention(d_model, num_heads, k)
        else:
            self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.ffn = models.Sequential([layers.Dense(ffn_dim, activation='relu'), layers.Dense(d_model)])
        self.layernorm1, self.layernorm2 = layers.LayerNormalization(), layers.LayerNormalization()
        self.dropout1, self.dropout2 = layers.Dropout(dropout_rate), layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        if self.k:
            attn_output = self.attn(x, training=training)
        else:
            attn_output = self.attn(x, x, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        x = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(x, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        x = self.layernorm2(x + ffn_output)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "num_heads": self.num_heads, "ffn_dim": self.ffn_dim, "dropout_rate": self.dropout_rate})
        return config
    
    def create_transformer_model(num_features):
        inputs = layers.Input(shape=(num_features,))
        x = FeatureEmbedding(128)(inputs)
        pos_encoding = tf.Variable(tf.random.normal([1, num_features, 128], stddev=0.02), trainable=True)
        x = x + pos_encoding
        for _ in range(2): x = TransformerEncoderBlock(128, 4, 256, 0.1)(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.1)(x)
        outputs = layers.Dense(1)(x)
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "num_heads": self.num_heads, "ffn_dim": self.ffn_dim, "dropout_rate": self.dropout_rate, "k": self.k})
        return config
