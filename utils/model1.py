import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Embedding, LSTM, AdditiveAttention
from tensorflow.keras.models import Model

# Paso 1: Crear una CNN simple para extraer características de imágenes
def create_cnn_model(input_shape=(224, 224, 3)):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(256, activation='relu')(x)  # Vector de características
    return Model(inputs, outputs, name="CNN_Model")

# Paso 2: Crear la LSTM con mecanismo de atención
def create_captioning_model(vocab_size, embedding_dim=256, units=256, max_length=20):
    # Entrada de características de la CNN
    image_features = Input(shape=(256,), name="Image_Features")

    # Entrada de texto (secuencias de palabras)
    caption_input = Input(shape=(max_length,), name="Caption_Input")
    
    # Embedding para la entrada de texto
    caption_embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)(caption_input)

    # LSTM para procesar las secuencias de texto
    lstm_output, state_h, state_c = LSTM(units, return_sequences=True, return_state=True)(caption_embedding)

    # Mecanismo de Atención
    attention = AdditiveAttention()([tf.expand_dims(image_features, axis=1), lstm_output])

    # Concatenar salida de atención y salida de la LSTM
    combined = tf.concat([attention, lstm_output], axis=-1)

    # Predicción final
    outputs = Dense(vocab_size, activation="softmax")(combined)

    return Model(inputs=[image_features, caption_input], outputs=outputs, name="Image_Captioning_Model")

# Crear la CNN y el modelo de captioning
cnn_model = create_cnn_model()
captioning_model = create_captioning_model(vocab_size=5000, embedding_dim=256, units=256, max_length=20)

# Compilar y mostrar los modelos
cnn_model.summary()
captioning_model.summary()

# Paso 3: Conectar la CNN y la LSTM
# Aquí se asume que las imágenes ya están preprocesadas y listas para alimentar a la CNN
# Extraer características de las imágenes
# image_features = cnn_model.predict(images)  # Usar estas características para alimentar al modelo de LSTM
