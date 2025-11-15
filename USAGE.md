# RFP Agent Usage Guide

## Getting Started

### Step 1: Setup Environment

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure your API key:
```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### Step 2: Build Your Knowledge Base

The knowledge base stores historical RFP answers that the AI will reference when generating new responses.

#### From a CSV file:
```bash
python -m rfp_agent.main build-kb --csv your_historical_answers.csv
```

Your CSV should have these columns:
- `question`: The RFP question
- `answer`: Your historical answer
- Optional: `category`, `client`, `date`, etc. (stored as metadata)

Example CSV format:
```csv
question,answer,category
"What is your data retention policy?","We retain data for 7 years...",Compliance
"Do you support SSO?","Yes, we support SAML 2.0...",Security
```

#### Test with example data:
```bash
python -m rfp_agent.main build-kb --csv example_kb.csv
```

### Step 3: Generate Answers for New RFPs

Once your knowledge base is ready, generate answers for new RFPs:

```bash
python -m rfp_agent.main generate new_rfp.csv -o answers.csv
```

The RFP CSV should have a `question` column:
```csv
question
"What are your backup procedures?"
"Do you offer multi-factor authentication?"
```

#### Test with example data:
```bash
python -m rfp_agent.main generate example_rfp.csv -o example_answers.csv
```

## Advanced Usage

### Custom Column Names

If your CSV uses different column names:

**Build KB:**
```bash
python -m rfp_agent.main build-kb --csv data.csv \
  --question-col "RFP Question" \
  --answer-col "Our Answer"
```

**Generate:**
```bash
python -m rfp_agent.main generate rfp.csv \
  --question-col "Questions" \
  -o output.csv
```

### Configuration Options

Edit `.env` to customize:

```env
GOOGLE_API_KEY=your_key_here
KB_PATH=knowledge_base           # Where KB files are stored
DATA_PATH=data                   # Customer data directory
GEMINI_MODEL=gemini-1.5-pro     # AI model to use
API_DELAY=1.0                    # Delay between API calls
```

## Workflow Examples

### Complete Workflow Example

```bash
# 1. Build knowledge base from historical data
python -m rfp_agent.main build-kb --csv historical_rfp_answers.csv

# 2. Generate answers for new RFP
python -m rfp_agent.main generate new_rfp_questions.csv -o new_rfp_answers.csv

# 3. Review the generated answers in new_rfp_answers.csv
```

### Adding More Historical Data

You can run `build-kb` multiple times to add more data:

```bash
# Add Q1 historical answers
python -m rfp_agent.main build-kb --csv q1_answers.csv

# Add Q2 historical answers
python -m rfp_agent.main build-kb --csv q2_answers.csv

# Knowledge base now contains both Q1 and Q2 answers
```

## Tips for Best Results

1. **Quality Knowledge Base**: The better your historical answers, the better the generated responses
2. **Categorize Questions**: Use metadata columns (category, tags) to organize your KB
3. **Review Generated Answers**: Always review AI-generated answers before submitting
4. **Iterative Improvement**: Add good generated answers back to your KB for future use
5. **API Rate Limits**: Adjust `API_DELAY` if you hit rate limits

## Troubleshooting

### "GOOGLE_API_KEY not set"
- Make sure you created `.env` file from `.env.example`
- Add your Google API key to the `.env` file

### "Column 'question' not found in CSV"
- Check your CSV has the right column names
- Use `--question-col` to specify the correct column name

### API rate limits
- Increase `API_DELAY` in `.env` (e.g., `API_DELAY=2.0`)
- Process RFPs in smaller batches

### File upload errors
- Check your internet connection
- Verify your Google API key has the necessary permissions
- Large KB files may take time to upload

## Output Format

Generated answers CSV contains:
- `id`: Question ID (Q1, Q2, etc.)
- `question`: The original question
- `answer`: AI-generated answer

Example output:
```csv
id,question,answer
Q1,"What are your backup procedures?","We implement automated daily backups..."
Q2,"Do you offer MFA?","Yes, we support multiple MFA methods..."
```

## Next Steps

After generating answers:
1. Review all generated responses
2. Edit as needed for accuracy and tone
3. Add approved answers to your knowledge base
4. Submit your completed RFP responses
