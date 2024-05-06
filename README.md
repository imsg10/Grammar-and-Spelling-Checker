# Advanced Grammar, Spelling, and Sentiment Checker

This Streamlit web application provides advanced text analysis functionalities, including grammar and spelling correction, sentiment analysis, text summarization, and word frequency analysis.

## Features

- **Grammar and Spelling Correction:** Utilizes TextBlob and spaCy to correct grammar and spelling errors in the input text.
- **Sentiment Analysis:** Analyzes the sentiment of the input text using the VADER sentiment analysis tool.
- **Text Summarization:** Summarizes the input text to provide a concise overview of the main points.
- **Word Frequency Analysis:** Calculates the frequency of each word in the input text and displays the top 5 most common meaningful words.

## Usage

1. Enter your text in the provided text area.
2. Click the "Check" button to perform the analysis.
3. View the corrected text, sentiment analysis results, summarized text, and word frequency analysis.

## Installation

To run this Streamlit app locally, follow these steps:

1. Install the required Python packages:
   ```bash
   pip install streamlit spacy textblob nltk
   ```
2. Download the required NLTK data:
   ```bash
   python -m nltk.downloader punkt vader_lexicon
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Contributors

- Samriddhi Gupta

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to customize the content based on your specific project details and preferences.
