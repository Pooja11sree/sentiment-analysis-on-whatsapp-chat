# 💬 Sentiment Analysis on WhatsApp Chat

Analyze your WhatsApp chat messages to discover the emotional tone of your conversations using Natural Language Processing (NLP). This project uses Python to perform **sentiment analysis** and **link extraction** from an exported WhatsApp `.txt` chat file, providing visual insights into how the conversations flow over time.

---

## 📌 Project Overview
This project aims to:
- Perform sentiment analysis (Positive, Neutral, Negative) on each message
- Visualize the sentiment distribution using graphs
- Extract shared URLs and find the most frequently used links
- Provide overall insights about the chat in a user-friendly way

It’s ideal for gaining insights into personal or group chats, observing mood trends, or just exploring your WhatsApp data for fun and learning.

## 🔍 Key Features
✅ **Sentiment Detection**  
Classifies each message into Positive, Negative, or Neutral using VADER sentiment analyzer.

✅ **Chat Parsing**  
Reads exported `.txt` files and cleans messages for processing.

✅ **URL Extraction**  
Identifies all shared URLs and ranks the most frequently used ones.

✅ **Visualizations**  
Generates clean graphs using Seaborn & Matplotlib for:
- Sentiment distribution
- Timeline trends (optional)
- Link usage statistics

✅ **Simple & Clean Output**  
Readable summaries and visual files saved to a folder for review.

## 🧠 Technologies Used

| Tool/Library     | Purpose                              |
|------------------|--------------------------------------|
| `Python`         | Core programming language            |
| `pandas`         | Data manipulation and analysis       |
| `nltk` (VADER)   | Natural Language Processing & Sentiment Analysis |
| `Seaborn`        | Data visualization                   |
| `Matplotlib`     | Plotting charts                      |
| `re`             | Regular expressions for text parsing |


💡Future Enhancements
 Add word cloud generation
 Show most active users in group chats
 Track sentiment over time (timeline)
 Build a simple GUI interface for easier file upload


