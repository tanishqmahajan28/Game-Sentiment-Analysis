# Twitter Sentiment Analysis on Games 🎮

This project focuses on performing **Sentiment Analysis** on Twitter comments related to various games using **Natural Language Processing (NLP)** and **Deep Learning** models like **RNN** and **LSTM**. The goal is to classify the sentiment of users' comments (e.g., positive, negative) towards different games based on the feedback provided in tweets.

## Project Overview 📄

The project involves:
- **Data Collection**: We collected Twitter comments about various games such as "Borderlands", "Nvidia", and others.
- **Data Preprocessing**: Applied text cleaning techniques such as lowercasing, removing numbers, punctuation, and stopwords, followed by tokenization and padding of sequences.
- **Modeling**: Implemented deep learning models, including RNN and LSTM, to capture the sequential nature of text and predict the sentiment (Positive or Negative).
- **Evaluation**: The model's performance was evaluated using accuracy and loss metrics.

## Key Features 🚀

- **Natural Language Processing (NLP)**: Used to preprocess and clean the text data.
- **Deep Learning Models**: Implemented RNN and LSTM to handle sequential dependencies in textual data.
- **Sentiment Prediction**: The model predicts whether a given comment is **Positive** or **Negative**.

## Dataset 📊

The dataset consists of Twitter comments for various games, with columns for:
- `Sr No`: Serial number of the tweet.
- `Game`: Name of the game being discussed.
- `Feedback`: Sentiment label, either **Positive** or **Negative**.
- `Comment`: The actual tweet text.

### Sample Data

| Sr No | Game        | Feedback | Comment                                      |
|-------|-------------|----------|----------------------------------------------|
| 2401  | Borderlands | Positive | im getting on borderlands and I will murder… |

## Model Architecture 🧠

We used **Recurrent Neural Networks (RNN)** and **Long Short-Term Memory (LSTM)** models to train on the sequential text data.

1. **Embedding Layer**: Converts words into vector representations.
2. **RNN/LSTM Layers**: Capture temporal dependencies in the text.
3. **Dense Layer**: Output layer for classification.

## Libraries and Tools 🛠️

- **TensorFlow/Keras**: For building and training the RNN/LSTM models.
- **Numpy & Pandas**: For data manipulation.
- **Sklearn**: For splitting the dataset and encoding labels.
- **Gradio**: For creating an interactive web interface to predict sentiment.

## Installation ⚙️

To run this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/twitter-sentiment-analysis.git

2. Install the required dependancies:
    ```bash
   pip install tensorflow gradio pandas numpy scikit-learn

## Usage 💻
1. Preprocess the data and train the model by running the Jupyter notebook or Python script.
2. Use the Gradio interface for real-time sentiment prediction:
   ```bash
   # Run the Gradio interface
   iface.launch()
3. Enter any text to predict its sentiment.

## Results 📈
The model achieves high accuracy in predicting whether a comment is Positive or Negative. It effectively captures the sentiment of user feedback on games using deep learning techniques like RNN and LSTM.

- Example Input: "I'm loving this new game!"
- Predicted Output: Positive

## Conclusion 🏁

This project demonstrates the power of NLP and deep learning models like RNN and LSTM in accurately predicting sentiment from Twitter comments related to games. By utilizing text preprocessing techniques and sequential neural networks, we were able to classify user feedback effectively into positive or negative sentiments.

## Contact 📞

For any queries or suggestions, please contact Tanishq Mahajan at trmahajan28@gmail.com.

