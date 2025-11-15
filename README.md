# RFP Agent - Semi-Automated RFP Answer Engine

A Python-based tool for semi-automated RFP (Request for Proposal) response generation. This system builds a canonical knowledge base from historical RFP answers and uses Google's Gemini AI (with File Search) to batch-generate pragmatic, SaaS-focused answers for new RFP CSV files.

## Features

- 📚 **Knowledge Base Builder**: Organize historical RFP answers into a searchable knowledge base
- 🤖 **Gemini Integration**: Leverage Google's Gemini AI with File Search for intelligent answer generation
- 📊 **CSV Processing**: Batch process RFP questions from CSV files
- 🔒 **Privacy First**: Customer data and KB files are gitignored by default
- ⚙️ **Configurable**: Easy configuration via environment variables
- 🚀 **SaaS-Focused**: Optimized for generating professional, pragmatic SaaS responses

## Installation

1. Clone the repository:
```bash
git clone https://github.com/rdwornik/rfp-agent-cognitive-planning.git
cd rfp-agent-cognitive-planning
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up configuration:
```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

## Quick Start

### 1. Build Knowledge Base

First, populate your knowledge base with historical RFP Q&A pairs:

```bash
python -m rfp_agent.main build-kb --csv example_kb.csv
```

This imports historical questions and answers into the knowledge base at `knowledge_base/`.

### 2. Generate Answers

Generate answers for a new RFP:

```bash
python -m rfp_agent.main generate example_rfp.csv -o answers.csv
```

This will:
1. Load questions from the CSV file
2. Upload your knowledge base to Gemini
3. Generate answers for each question
4. Save results to the output file

## Usage

### Building the Knowledge Base

Import from CSV:
```bash
python -m rfp_agent.main build-kb --csv historical_answers.csv \
  --question-col "Question" \
  --answer-col "Answer"
```

The CSV should have columns for questions and answers. Additional columns will be stored as metadata.

### Generating Answers

Basic usage:
```bash
python -m rfp_agent.main generate new_rfp.csv
```

With custom output:
```bash
python -m rfp_agent.main generate new_rfp.csv -o responses.csv
```

Specify question column:
```bash
python -m rfp_agent.main generate rfp.csv --question-col "RFP Question"
```

## Configuration

Configure the application using environment variables in `.env`:

```env
# Required
GOOGLE_API_KEY=your_google_api_key

# Optional
KB_PATH=knowledge_base          # Knowledge base directory
DATA_PATH=data                  # Customer data directory
GEMINI_MODEL=gemini-1.5-pro    # Gemini model to use
BATCH_SIZE=10                   # Questions per batch
API_DELAY=1.0                   # Delay between API calls (seconds)
```

## Project Structure

```
rfp-agent-cognitive-planning/
├── rfp_agent/              # Main package
│   ├── __init__.py
│   ├── main.py            # CLI entry point
│   ├── config.py          # Configuration management
│   ├── kb_builder.py      # Knowledge base builder
│   ├── rfp_processor.py   # RFP CSV processor
│   └── gemini_agent.py    # Gemini AI integration
├── knowledge_base/        # KB files (gitignored)
├── data/                  # Customer data (gitignored)
├── example_kb.csv         # Example knowledge base
├── example_rfp.csv        # Example RFP questions
├── requirements.txt       # Python dependencies
├── .env.example          # Example environment config
└── README.md             # This file
```

## CSV Format

### Knowledge Base CSV

The historical answers CSV should have at minimum:
- `question`: The RFP question
- `answer`: The historical answer

Optional columns (stored as metadata):
- `category`: Question category
- `client`: Client name
- `date`: Date of answer
- Any other relevant metadata

Example:
```csv
question,answer,category
"What is your data retention policy?","We retain data for 7 years...",Compliance
"Do you support SSO?","Yes, we support SAML 2.0...",Security
```

### RFP Input CSV

The RFP questions CSV should have at minimum:
- `question`: The RFP question

Example:
```csv
question
"What are your backup procedures?"
"Do you offer MFA?"
```

### Output CSV

The generated answers CSV will contain:
- `id`: Question ID
- `question`: The original question
- `answer`: Generated answer

## Development

### Running Tests

(Tests can be added as the project grows)

```bash
pytest tests/
```

### Code Style

```bash
# Format code
black rfp_agent/

# Lint
pylint rfp_agent/
```

## Security & Privacy

- Customer data and KB files are automatically gitignored
- API keys are stored in `.env` (also gitignored)
- No sensitive data is committed to the repository

## API Usage & Costs

This tool uses Google's Gemini API which has usage costs. Monitor your usage:
- File uploads count toward storage limits
- Each API call for answer generation consumes tokens
- Consider the `API_DELAY` setting to manage rate limits

## Limitations

- Requires valid Google API key with Gemini access
- Answer quality depends on knowledge base quality
- Large knowledge bases may take time to upload
- Subject to Gemini API rate limits and quotas

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For issues and questions:
- Open an issue on GitHub
- Check existing issues for solutions

## Roadmap

Potential future enhancements:
- [ ] Support for multiple document formats (PDF, DOCX)
- [ ] Advanced KB organization with categories/tags
- [ ] Answer quality scoring
- [ ] Interactive review interface
- [ ] Multi-language support
- [ ] Integration with other LLMs
- [ ] Caching for common questions