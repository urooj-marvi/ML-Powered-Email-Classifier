# 📧 ML-Powered Email Classifier Dashboard

An interactive web application for automatically categorizing incoming emails using machine learning models. This dashboard provides a user-friendly interface for email classification, model performance analysis, and data insights.

## 🚀 Features

- **📊 Real-time Email Classification** - Paste any email and get instant classification
- **🤖 Multiple ML Models** - SVM (99.86% accuracy) and Naive Bayes (98.95% accuracy)
- **📈 Interactive Visualizations** - Beautiful charts and performance metrics
- **🎨 Modern UI** - Responsive design with color-coded results
- **📋 Comprehensive Analysis** - Data insights and model performance comparison

## 📧 Email Categories

- **📧 Work**: Regular work-related emails, meetings, updates
- **⚠️ Important**: Critical emails requiring immediate attention
- **🚫 Spam**: Unwanted or promotional emails

## 🛠️ Installation

### Option 1: Local Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/email-classifier-dashboard.git
   cd email-classifier-dashboard
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements_dashboard.txt
   ```
   
   Or use the setup script:
   ```bash
   python setup.py
   ```

3. **Train the models (if not included):**
   ```bash
   python train_models.py
   ```

### Option 2: Docker Installation

1. **Using Docker Compose (Recommended):**
   ```bash
   docker-compose up --build
   ```

2. **Using Docker directly:**
   ```bash
   docker build -t email-classifier-dashboard .
   docker run -p 8501:8501 email-classifier-dashboard
   ```

### Option 3: Cloud Deployment

The dashboard is compatible with:
- **Streamlit Cloud**: Deploy directly from GitHub
- **Heroku**: Use the provided Dockerfile
- **AWS/GCP/Azure**: Use Docker containers

## 🚀 Usage

1. **Run the dashboard:**
   ```bash
   streamlit run email_classifier_dashboard.py
   ```

2. **Open your browser** to the URL shown (usually `http://localhost:8501`)

3. **Navigate through the dashboard:**
   - **📊 Overview**: General statistics and data distribution
   - **🔍 Email Classifier**: Main classification interface
   - **📈 Model Performance**: Model comparison and metrics
   - **📋 Data Analysis**: Detailed data insights

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| SVM (TF-IDF) | 99.86% | 99.82% | 99.20% | 99.51% |
| Naive Bayes (TF-IDF) | 98.95% | 97.59% | 87.82% | 91.35% |

## 🎯 Use Cases

### For Organizations
- **Email Management**: Automatically categorize incoming emails
- **Priority Filtering**: Identify important emails requiring immediate attention
- **Spam Detection**: Filter out unwanted promotional emails
- **Workflow Optimization**: Route emails to appropriate teams/departments

### For Individuals
- **Inbox Organization**: Automatically sort emails by importance
- **Time Management**: Focus on important emails first
- **Spam Protection**: Reduce time spent on unwanted emails

## 📁 Project Structure

```
email-classifier-dashboard/
├── 📄 README.md                    # This file
├── 📄 requirements_dashboard.txt   # Python dependencies
├── 🐍 email_classifier_dashboard.py # Main dashboard application
├── 🐍 train_models.py             # Model training script
├── 🐍 simple_test.py              # Component testing script
├── 📊 emails_cleaned.csv          # Main dataset (10,947 emails)
├── 🤖 tfidf_vectorizer.pkl        # Pre-trained TF-IDF vectorizer
├── 🤖 naive_bayes_tfidf.pkl       # Pre-trained Naive Bayes model
├── 🤖 svm_tfidf.pkl               # Pre-trained SVM model
└── 📚 docs/                       # Documentation folder
    ├── README_Dashboard.md        # Detailed documentation
    └── DASHBOARD_SUMMARY.md       # Project summary
```

## 🧪 Testing

Run the test script to verify all components work correctly:

```bash
python simple_test.py
```

Expected output:
```
🧪 Testing Email Classifier Dashboard Components
==================================================
1. Testing data loading...
   ✅ Data loaded successfully: (10947, 5)
   ✅ Categories: ['Spam' 'Important' 'Work']

2. Testing model loading...
   ✅ TF-IDF vectorizer loaded
   ✅ Naive Bayes model loaded
   ✅ SVM model loaded

3. Testing prediction...
   ✅ Prediction test successful: 'Hi team, please review the quarterly report by Fri...' -> Work

==================================================
🎉 Testing completed!
```

## 🛠️ Technical Stack

- **Frontend**: Streamlit (Python web framework)
- **ML**: Scikit-learn (SVM, Naive Bayes, TF-IDF)
- **Visualization**: Plotly, Seaborn, Matplotlib
- **NLP**: NLTK, WordCloud
- **Data Processing**: Pandas, NumPy

## 🔧 Machine Learning Pipeline

1. **Text Preprocessing**: Convert to lowercase, remove special characters
2. **Feature Extraction**: TF-IDF vectorization with 5000 features
3. **Model Training**: Support Vector Machine and Multinomial Naive Bayes
4. **Validation**: Train-test split with stratification

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔧 Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'plotly'**
   ```bash
   pip install plotly seaborn wordcloud nltk
   ```

2. **NLTK Data Missing**
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

3. **Port Already in Use**
   ```bash
   streamlit run email_classifier_dashboard.py --server.port 8502
   ```

4. **Model Files Missing**
   ```bash
   python train_models.py
   ```

### Environment Issues

- **Windows**: Use PowerShell or Command Prompt
- **macOS/Linux**: Use Terminal
- **Cloud/Container**: Use the provided Dockerfile

## 📞 Support

If you have any questions or need support, please:
- Open an issue on GitHub
- Check the documentation in the `docs/` folder
- Review the testing results
- Run `python simple_test.py` to verify installation

## 🔮 Future Enhancements

- Real-time email server integration
- Custom category definitions
- Model retraining interface
- API endpoints for programmatic access
- Multi-language support

---

## 🎉 Acknowledgments

- Built with ❤️ using Streamlit and Scikit-learn
- Dataset includes emails from various sources for comprehensive training
- Special thanks to the open-source community for amazing tools and libraries

**Happy Email Classification! 📧✨**

## Deployment
Streamlit code: https://ml-powered-email-classifier-cv84hbgucmtuwhquknv3rz.streamlit.app/
---

⭐ **If you find this project helpful, please give it a star!**
